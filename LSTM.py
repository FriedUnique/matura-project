from typing import List
import numpy as np

# https://medium.com/@aidangomez/let-s-do-this-f9b699de31d9


def Sigmoid(x): 
    return 1. / (1 + np.exp(-x))

def ablSigmoid(values): 
    return values*(1-values)

def ablTanh(values): 
    return 1. - values ** 2

def UniformXavier(hiddenSize, inputSize, finalSize: tuple):
    limit = np.sqrt(6.0 / (hiddenSize + inputSize))
    return np.random.uniform(-limit, limit, size=finalSize)


def InitLstmParams(hiddenSize, inputSize):
    concat = hiddenSize + inputSize
    limit = np.sqrt(6.0 / (hiddenSize + inputSize))

    wa = np.random.uniform(-limit, limit, size = (hiddenSize, concat))
    wi = np.random.uniform(-limit, limit, size = (hiddenSize, concat))
    wf = np.random.uniform(-limit, limit, size = (hiddenSize, concat))
    wo = np.random.uniform(-limit, limit, size = (hiddenSize, concat))

    ba = np.random.uniform(-limit, limit, size = (hiddenSize))
    bi = np.random.uniform(-limit, limit, size = (hiddenSize))
    bf = np.random.uniform(-limit, limit, size = (hiddenSize))
    bo = np.random.uniform(-limit, limit, size = (hiddenSize))

    return wa, wi, wf, wo, ba, bi, bf, bo


class Lstm_Cell:
    def __init__(self, params, hiddenSize, inputSize):       
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.wc, self.wi, self.wf, self.wo, self.bc, self.bi, self.bf, self.bo = params

        # states
        self.a = np.zeros(hiddenSize) # input activation
        self.i = np.zeros(hiddenSize) # input gate
        self.f = np.zeros(hiddenSize) # forget gate
        self.o = np.zeros(hiddenSize) # output gate

        self.s = np.zeros(hiddenSize) # state
        self.h = np.zeros(hiddenSize) # output

        self.s_prev = np.zeros_like(self.s) # t-1
        self.h_prev = np.zeros_like(self.h) # t-1

        self.diff_h = np.zeros_like(self.h) # t+1
        self.diff_s = np.zeros_like(self.s) # t+1

        # input for this cell (current input), stored for the back propagation
        self.xc = None

    def Forward(self, x, s_prev = None, h_prev = None):
        if s_prev is None: s_prev = np.zeros_like(self.s)
        if h_prev is None: h_prev = np.zeros_like(self.h)
        # save data for use in backprop
        self.s_prev = s_prev
        self.h_prev = h_prev

        # mergin input and prev hidden state, because it is easier and is better for performance. this is why there 
        # only are 4 weights and not 8

        xc = np.hstack((x,  h_prev))
        # maybe not dot but rather inner product
        self.a = np.tanh(np.dot(self.wc, xc) + self.bc)
        self.i = Sigmoid(np.dot(self.wi, xc) + self.bi)

        print(self.i)
        print(x.shape)
        print(h_prev.shape)
        print(xc.shape)
        print(self.wi.shape)
        print((np.dot(self.wi, xc)+self.bi).shape)
        print()

        self.f = Sigmoid(np.dot(self.wf, xc) + self.bf)
        self.o = Sigmoid(np.dot(self.wo, xc) + self.bo)


        self.s = self.s_prev * self.f + self.a * self.i
        self.h = np.tanh(self.s) * self.o

        self.xc = xc

    def Backward(self, diff_h, future_d_s):
        # diff_h = d_out

        d_s = diff_h * self.o * ablTanh(self.s) + future_d_s * self.f # f+1?
        d_a = d_s * self.i * ablTanh(self.a)
        d_i = d_s * self.a * ablSigmoid(self.i)
        d_f = d_s * self.prev_s * ablSigmoid(self.f)
        d_o = diff_h * np.tanh(self.s) * ablSigmoid(self.o)

        # outer returns a matrix

        wa_grad = np.outer(d_a, self.xc)
        wi_grad = np.outer(d_i, self.xc)
        wf_grad = np.outer(d_f, self.xc)
        wo_grad = np.outer(d_o, self.xc)

        ba_grad = d_a   
        bi_grad = d_i
        bf_grad = d_f       
        bo_grad = d_o

        # Gradients with respect to input:
        dxc = np.zeros_like(self.xc)
        dxc += np.dot(self.wi.T, d_i)
        dxc += np.dot(self.wf.T, d_f)
        dxc += np.dot(self.wo.T, d_o)
        dxc += np.dot(self.wc.T, d_a)

        self.diff_s = d_s
        self.diff_h = dxc[self.inputSize:]

        return wa_grad, wi_grad, wf_grad, wo_grad, ba_grad, bi_grad, bf_grad, bo_grad
  

def ClipGradients(grads, maxNorm):
    return [np.clip(g, -maxNorm, maxNorm) for g in grads]
  
def UpdateParams(grads, params, lr=0.001):
    for param, grad in zip(params, grads):
        param -= lr * grad

    return params




def Loss(yLabel, yPred):
    # squared loss error
    return (yPred[0] - yLabel) ** 2

def Cost(yLabel, yPred):
    return 2 * (yPred[0] - yLabel)

def Softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class Lstm():
    def __init__(self, hiddenSize, inputSize):
        self.hiddenSize = hiddenSize
        self.inputSize = inputSize
        
        self.params = InitLstmParams(hiddenSize, inputSize)
        self.cells: List[Lstm_Cell] = []
        self.recurentInputs = [] # inputs


    def Train(self, xList, yList, lr=0.01):
        for x in xList:
            self.recurentInputs.append(x)
            # check if there is a cell for new input data
            if len(self.recurentInputs) > len(self.cells):
                self.cells.append(Lstm_Cell(self.params, self.hiddenSize, self.inputSize))

            # Forward; if there is no previous cell, then there is no previous cell or hidden state
            idx = len(self.recurentInputs) - 1
            if idx == 0:
                self.cells[idx].Forward(x)
            else:
                s_prev = self.cells[idx - 1].s
                h_prev = self.cells[idx - 1].h
                self.cells[idx].Forward(x, s_prev, h_prev)

        # Backward
        idx = len(self.recurentInputs) - 1

        loss = Loss(yList[idx], self.cells[idx].h)
        diff_h = Cost(yList[idx], self.cells[idx].h)
        diff_s = np.zeros(self.hiddenSize)

        gradients = self.cells[idx].Backward(diff_h, diff_s)
        self.params = UpdateParams(ClipGradients(gradients, 1), self.params, lr)
        idx -= 1

        while idx >= 0:

            # yhat = self.cells[idx].h

            # not the loss whudiasnoipdhfioahf
            loss += Loss(yList[idx], self.cells[idx].h)




            #dout  = deltaT + deltaOut
            diff_h = Cost(yList[idx], self.cells[idx].h) 
            diff_h -= self.cells[idx+1].diff_h # changed sign

            diff_s = self.cells[idx+1].diff_s # future_s

            gradients = self.cells[idx].Backward(diff_h, diff_s)
            self.params = UpdateParams(ClipGradients(gradients, 1), self.params, lr)
            idx -= 1 

        self.recurentInputs = []
        self.cells = [] # ?
        predictions = [self.cells[i].h[0] for i in range(len(yList))]

        return loss, predictions
  



def Main():
    np.random.seed(0)

    hiddenSize = 100
    inputDim = 50 # inputSize number of feautres
    lstm = Lstm(hiddenSize, inputDim)
    yLabels = [-0.011, 0.1, 0.04, -0.08]
    inputs = [np.random.uniform(0, 10, inputDim) for _ in yLabels]

    for i in range(10):
        loss, predictions = lstm.Train(inputs, yLabels)

        #print(loss, predictions)


"""
 #dWs = self.o * prevH_grad + prevS_grad

        dWo = diff_h * self.s * ablSigmoid(self.o)
        dWc = self.i * (diff_h + future_s) * ablTanh(self.a)
        dWi = future_s * self.a * ablSigmoid(self.i)
        dWf = future_s * self.s_prev * ablSigmoid(self.f)

        #dWo = (self.s * prevH_grad) * dSigmoid(self.o)
        #dWi = (self.c * dWs) * dSigmoid(self.i)
        #dWc = (self.i * dWs) * dTanh(self.c)
        #dWf = (self.s_prev * dWs) * dSigmoid(self.f)

        wi_grad = np.outer(dWi, self.xc)
        wf_grad = np.outer(dWf, self.xc)
        wo_grad = np.outer(dWo, self.xc)
        wc_grad = np.outer(dWc, self.xc)

        bi_grad = dWi
        bf_grad = dWf       
        bo_grad = dWo
        bc_grad = dWc   

        # Gradients with respect to input:
        dxc = np.zeros_like(self.xc)
        dxc += np.dot(self.wi.T, dWi)
        dxc += np.dot(self.wf.T, dWf)
        dxc += np.dot(self.wo.T, dWo)
        dxc += np.dot(self.wc.T, dWc)

        self.diff_s = future_s * self.f
        self.diff_h = dxc[self.inputSize:]


"""