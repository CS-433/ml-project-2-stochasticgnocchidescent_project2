import torch
import numpy as np
import time
import copy

# class implementing the page algorithm, plainly as described in the original paper
# call .step(t) to perform the t-th optimization step
class PlainPageDescent():

    def __init__(self, model, loss_func, eta, coin_tosses, batches, x, y, T, store_seq = False):

        self.model = model
        self.loss_func = loss_func

        self.prev_g = None # stores previous iteration's update to weights, if needed

        self.prev_grad = None # store the previous iteration's gradient, if needed

        self.eta = eta

        self.coin_tosses = coin_tosses # list containing T heads or tails, according to which we perform either of the two optimization step as described in the algorithms
        self.batches = batches # contains indices of datapoints in each batch
        self.x = x # training x
        self.y = y # training labels
        self.T = T # number of iterations

        if store_seq:  # used for randomly returning any of the weights in the history
            self.parameters_seq = []
        else:
            self.parameters_seq = None
    
    def step(self, t):
        g = None
        if self.coin_tosses[t]: # heads
            bx = self.x[self.batches[t]]
            by = self.y[self.batches[t]]

            # forward pass
            y_pred = self.model(bx)
            loss = self.loss_func(y_pred, by)             

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
                bx = self.x[self.batches[t+1]]
                by = self.y[self.batches[t+1]]

                # forward pass
                y_pred = self.model(bx)
                loss = self.loss_func(y_pred, by)
                

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

            # forward pass on this iteration's batch with this iteration's weights
            bx = self.x[self.batches[t]]
            by = self.y[self.batches[t]]

            y_pred = self.model(bx)
            loss = self.loss_func(y_pred, by)

            
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
                bx = self.x[self.batches[t]]
                by = self.y[self.batches[t]]

                # forward pass
                y_pred = self.model(bx)
                loss = self.loss_func(y_pred, by)
                

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

# trains the given neur_net on the given training set, with the plain version of PAGE
# takes train train set x,y, and test set test_x,test_y for computing test accuracy.
# neur_net is the model to optimize
# b is batch size
# eta is step size
# log_file_name is where to store csv log (see below for format of output), set to None if don't want it
# print log prints to console logged data
# print msgs prints nice messages to console
# as_sgd determines if we want to set p=1, i.e. if we want basically SGD
# T is the number of iterations
def train_with_plain_page(x, y, test_x, test_y, neur_net, loss_func, T, b, eta, log_file_name, save_log, print_log, print_msgs, as_sgd = False):
    n = len(y)

    b_prime = int(np.floor(np.sqrt(b))) # b' is computed according to algo
    p = b_prime/(b_prime + b) if not as_sgd else 1 # p is also computed according to algo, set to 1 if want this run to behave as SGD
    
    # generate coin tosses with bias p, first iteration needs always be heads, as per the algo
    coin_tosses = [np.random.rand() <= p for _ in range(T)]
    coin_tosses[0] = True
    
    # generate random batch indices
    batches = []
    for t in range(T):
        batch_indices = np.random.randint(0, n, size = (b if coin_tosses[t] else b_prime))
        batches.append(batch_indices)

    optimizer = PlainPageDescent(neur_net, loss_func, eta, coin_tosses, batches, x, y, T)

    log = []

    computed_gradients = 0
    num_logged_events = 0

    if print_msgs:
        print("Starting training...")
        print("--------------------")
    start_time = time.time()  
    prev_grads = 0 # used for logging purposes
    for t in range(T):
        start_step = time.time()

        optimizer.step(t)

        end_step = time.time()

        # we have just computed size_of_the_t-th_batch gradients
        computed_gradients += (b if coin_tosses[t] else b_prime)

        if computed_gradients - prev_grads >= n and (save_log or print_log): # if more than n gradients have been accumulated, log
            prev_grads = computed_gradients
            current_train_loss = loss_func(neur_net(x), y)
            current_test_accuracy = torch.sum((neur_net(test_x)).argmax(dim = 1) == test_y)/len(test_y)
            
            if save_log:
                log.append((t, end_step - start_time, computed_gradients, current_train_loss.data.item(), current_test_accuracy.data.item()))
            if print_log:
                print(f'{t}, {end_step - start_time}, {computed_gradients}, {current_train_loss.data.item():1.5f}, {current_test_accuracy.data.item():1.5f}')

    end_time = time.time()
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
    return log

# this is the same as train_with_plain_page, except we return a random weight from the training history, as per the algo in its plainest version
# note this works only with neur_nets.MLP network for MNIST dataset
# comments below show where this differs from train_with_plain_page
def train_with_plain_page_random_return(x, y, test_x, test_y, neur_net, loss_func, T, b, eta, cuda, log_file_name, save_log, print_log, print_msgs, as_sgd = False):
    n = len(y)

    b_prime = int(np.floor(np.sqrt(b)))
    p = b_prime/(b_prime + b) if not as_sgd else 1
    
    coin_tosses = [np.random.rand() <= p for _ in range(T)]
    coin_tosses[0] = True
    
    batches = []
    for t in range(T):
        batch_indices = np.random.randint(0, n, size = (b if coin_tosses[t] else b_prime))
        batches.append(batch_indices)

    # note the last parameter here is true
    optimizer = PlainPageDescent(neur_net, loss_func, eta, coin_tosses, batches, x, y, T, True)

    log = []

    computed_gradients = 0
    num_logged_events = 0

    if print_msgs:
        print("Starting training...")
        print("--------------------")
    start_time = time.time()  
    prev_grads = 0
    for t in range(T):
        start_step = time.time()

        optimizer.step(t)

        end_step = time.time()

        computed_gradients += (b if coin_tosses[t] else b_prime)
        import neur_nets

        if computed_gradients - prev_grads >= n and (save_log or print_log):
            # make a temporary NN
            temp_nn = neur_nets.MLP(784, 100, 10, cuda)

            # set it weights to be randomly selected from the history of weights of PAGE
            i = np.random.randint(len(optimizer.parameters_seq))
            l = list(optimizer.parameters_seq[i])
            d = temp_nn.state_dict()
            for i, k in enumerate(d.keys()):
                d[k] = l[i]
            temp_nn.load_state_dict(d)

            prev_grads = computed_gradients

            # compute loss and accuracy on this NN
            current_train_loss = loss_func(temp_nn(x), y)
            current_test_accuracy = torch.sum((temp_nn(test_x)).argmax(dim = 1) == test_y)/len(test_y)
            
            if save_log:
                log.append((t, end_step - start_time, computed_gradients, current_train_loss.data.item(), current_test_accuracy.data.item()))
            if print_log:
                print(f'{t},{end_step - start_time},{computed_gradients},{current_train_loss.data.item():1.5f},{current_test_accuracy.data.item():1.5f}')

    end_time = time.time()
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
    return log