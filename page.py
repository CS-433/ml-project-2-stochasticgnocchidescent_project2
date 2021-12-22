import torch
import numpy as np
import time
import copy

# class implementing the page algorithm in our second (practical) version (see report)
# call .step(t) to perform the t-th optimization step
class PageDescent():

    def __init__(self, model, loss_func, eta, coin_tosses, batches_x, batches_y, T, store_seq = False):

        self.model = model
        self.loss_func = loss_func

        self.prev_g = None # stores previous iteration's update to weights, if needed

        self.prev_grad = None # store the previous iteration's gradient, if needed

        self.eta = eta

        self.coin_tosses = coin_tosses # list containing T heads or tails, according to which we perform either of the two optimization step as described in the algorithms
        self.batches_x = batches_x # contains the T batches on x
        self.batches_y = batches_y # contains the T batches on the labels
        self.T = T # the number of iterations

        if store_seq: # used for randomly returning any of the weights in the history
            self.parameters_seq = []
        else:
            self.parameters_seq = None
    
    def step(self, t):
            g = None
            if self.coin_tosses[t]: # heads
                x = self.batches_x[t]
                y = self.batches_y[t]

                # forward pass
                y_pred = self.model(x)
                loss = self.loss_func(y_pred, y)             

                for p in self.model.parameters():
                    if p.grad != None:
                        p.grad.zero_()
                
                # backward pass, i.e. compute gradients
                loss.backward()

                # look-ahead: do if next iteration is tails
                if t < self.T-1 and not self.coin_tosses[t+1]:
                    # copy the just-computed gradients
                    g = [copy.deepcopy(p.grad) for p in self.model.parameters()]

                    # get next iteration's batch
                    x = self.batches_x[t+1]
                    y = self.batches_y[t+1]

                    # forward pass
                    y_pred = self.model(x)
                    loss = self.loss_func(y_pred, y)

                    for p in self.model.parameters():
                        if p.grad != None:
                            p.grad.zero_()

                    # backward pass
                    loss.backward()
                else:
                    # make non-deep low memory copy of just computed gradients
                    g = [p.grad for p in self.model.parameters()]
            else: # tails
                # initialize this iteration's update as the previous iteration's update minus the gradient of previous iteration's weights on this iteration's batch
                g = [self.prev_g[i] - p.grad for i,p in enumerate(self.model.parameters())]

                x = self.batches_x[t]
                y = self.batches_y[t]

                # forward pass on this iteration's batch with this iteration's weights
                y_pred = self.model(x)
                loss = self.loss_func(y_pred, y)

                
                for p in self.model.parameters():
                    if p.grad != None:
                        p.grad.zero_()
                
                # backward pass
                loss.backward()

                # add this gradient to the update, now update variable g is complete
                for i,p in enumerate(self.model.parameters()):
                    g[i] += p.grad
                
                # look-ahead: do if next iteration is tails
                if t < self.T-1 and not self.coin_tosses[t+1]:
                    # get next iteration's batch
                    x = self.batches_x[t+1]
                    y = self.batches_y[t+1]

                    # forward pass
                    y_pred = self.model(x)
                    loss = self.loss_func(y_pred, y)
                    

                    for p in self.model.parameters():
                        if p.grad != None:
                            p.grad.zero_()

                    # backward pass
                    loss.backward()
            
            # make non-deep copy of this iteration's update
            self.prev_g = g
            # update the weights
            for i,p in enumerate(self.model.parameters()):
                with torch.no_grad():
                    p -= self.eta*g[i]
            
            # store weights if want to
            if self.parameters_seq != None:
                self.parameters_seq.append([copy.deepcopy(p) for p in self.model.parameters()])

# given training set, batch sizes b and b', and probability p
# generate coin tosses for heads or tails with bias p.
# Then permute the training set and partition it in batches of correct size depending on heads or tails
def make_batches_epoch_style(x, y, b, b_prime, p):
    n = len(y)
    shuffle_indices = np.random.permutation(np.arange(n))

    x = x[shuffle_indices]
    y = y[shuffle_indices]

    coin_tosses = []

    batches_x = []
    batches_y = []

    offset = 0
    # continue as long as batches fit
    while offset < n-b:
        if offset == 0 or np.random.rand() <= p:
            coin_tosses.append(True)
            size = b
        else:
            coin_tosses.append(False)
            size = b_prime
        
        batches_x.append(x[offset:offset+size])
        batches_y.append(y[offset:offset+size])

        offset += size
    
    # if there is something left out, add it
    if offset < n-1:
        if offset == n - b:
            coin_tosses.append(True)
            size = b
            batches_x.append(x[offset:offset+size])
            batches_y.append(y[offset:offset+size])
        else:
            coin_tosses.append(False)
            batches_x.append(x[offset:n])
            batches_y.append(y[offset:n])
    
    return coin_tosses, batches_x, batches_y

# serves as a sub-routine to train_epoch_style
# basically runs an instantiation of PAGE for as many iterations as are needed to compute roughly n gradients.
# uses make_batches_epoch_style to randomly partition the training set with random sizes b, b'.
# print_msgs is for printing nice messages during execution
# for other parameters see train_epoch_style
def train_single_init_epoch_style(x, y, test_x, test_y, neur_net, loss_func, b, eta, cuda, log_file_name, save_log, print_log, print_msgs, as_sgd = False):
    if cuda:
        x = x.cuda()
        y = y.cuda()

    b_prime = int(np.floor(np.sqrt(b))) # b' is computed according to algo
    p = b_prime/(b_prime + b) if not as_sgd else 1 # p is also computed according to algo, set to 1 if want this run to behave as SGD

    coin_tosses, batches_x, batches_y = make_batches_epoch_style(x, y, b, b_prime, p)
    T = len(coin_tosses) # number of needed iterations

    optimizer = PageDescent(neur_net, loss_func, eta, coin_tosses, batches_x, batches_y, T)

    log = []

    computed_gradients = 0
    num_logged_events = 0

    if print_msgs:
        print("Starting training...")
        print("--------------------")
    start_time = time.time()  
    for t in range(T):
        start_step = time.time()

        optimizer.step(t)

        end_step = time.time()

        # we have just computed size_of_the_t-th_batch gradients
        computed_gradients += (b if coin_tosses[t] else b_prime)

    end_time = time.time()

    if save_log or print_log:
        current_train_loss = loss_func(neur_net(x), y)
        current_test_accuracy = torch.sum((neur_net(test_x)).argmax(dim = 1) == test_y)/len(test_y)
        
        if save_log:
            log.append((t, end_step - start_time, computed_gradients, current_train_loss.data.item(), current_test_accuracy.data.item()))
        if print_log:
            print(f'{t}, {end_step - start_time}, {computed_gradients}, {current_train_loss.data.item():1.5f}, {current_test_accuracy.data.item():1.5f}')

    if print_msgs:
        print("--------------------")
        print("Training terminated after", end_time - start_time, "seconds")

    if log_file_name != None:
        # for each log event, write the folliwing fields:
        # iteration, time elapsed since training started, number of computed gradients, training loss, test accuracy
        #
        # where each each of these values is relative to the event logging time
        log_file = open(log_file_name, "w")
        for event in log:
            t, time_elapsed, computed_gradients, loss, accuracy = event
            log_file.write(f'{t},{time_elapsed},{computed_gradients},{loss:1.5f},{accuracy:1.5f}\n')
        log_file.close()
    return T, log

# trains the given neur_net on the given training set, using practical epoch-based version of PAGE
# takes train train set x,y, and test set test_x,test_y for computing test accuracy.
# neur_net is the model to optimize
# cuda set to True enables cuda on training set
# b is batch size (should b << n, otherwise use plain PAGE)
# eta is step size
# log_file_name is where to store csv log (see below for format of output), set to None if don't want it
# print log prints to console logged data
# as_sgd determines if we want to set p=1, i.e. if we want basically SGD
# num_inits is the number of epochs
def train_epoch_style(x, y, test_x, test_y, neur_net, loss_func, num_inits, b, eta, cuda, log_file_name, print_log, as_sgd = False):
    if cuda:
        x = x.cuda()
        y = y.cuda()

    save_log = log_file_name != None

    log = []

    print("Starting training...")
    print("--------------------")
    start_time = time.time()
    computed_gradients = 0
    count_iters = 0

    # for num_inits times, call train_single_init_epoch_style
    for init in range(num_inits):
        start_init = time.time()
        T, partial_log = train_single_init_epoch_style(x, y, test_x, test_y, neur_net, loss_func, b, eta, cuda = False, log_file_name = None, save_log=save_log, print_log = print_log, print_msgs = False, as_sgd = as_sgd)
        
        if save_log:
            # store in the log the info contained in the partial log returned by this epoch
            # make minor counting adjustments
            temp_comp_grads = 0
            for e in partial_log:
                temp_comp_grads += e[2]
                log.append((e[0] + count_iters, e[1] + start_init - start_time, e[2] + computed_gradients, e[3], e[4]))
            computed_gradients += temp_comp_grads
        count_iters += T
        
    end_time = time.time()
    print("--------------------")
    print("Training terminated after", end_time - start_time, "seconds")

    if save_log:
        # for each log event, write the folliwing fields:
        # iteration, time elapsed since training started, number of computed gradients, training loss, test accuracy
        #
        # where each each of these values is relative to the event logging time
        log_file = open(log_file_name, "w")
        for event in log:
            t, time_elapsed, computed_gradients, loss, accuracy = event
            log_file.write(f'{t},{time_elapsed},{computed_gradients},{loss:1.5f},{accuracy:1.5f}\n')
        log_file.close()
