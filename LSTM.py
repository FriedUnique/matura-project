from typing import List
import numpy as np


def Sigmoid(x): 
    return 1. / (1 + np.exp(-x))

def dSigmoid(values): 
    return values*(1-values)

def dTanh(values): 
    return 1. - values ** 2


def InitLstmParams(hiddenSize, inputSize):
    concat_len = inputSize + hiddenSize

    wc = np.random.uniform(-0.1, 0.1, size = (hiddenSize, concat_len))
    wi = np.random.uniform(-0.1, 0.1, size = (hiddenSize, concat_len))
    wf = np.random.uniform(-0.1, 0.1, size = (hiddenSize, concat_len))
    wo = np.random.uniform(-0.1, 0.1, size = (hiddenSize, concat_len))

    bc = np.random.uniform(-0.1, 0.1, size = hiddenSize) 
    bi = np.random.uniform(-0.1, 0.1, size = hiddenSize) 
    bf = np.random.uniform(-0.1, 0.1, size = hiddenSize) 
    bo = np.random.uniform(-0.1, 0.1, size = hiddenSize) 

    return wc, wi, wf, wo, bc, bi, bf, bo


class Lstm_Cell:
    def __init__(self, params, hiddenSize, inputSize):       
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.wc, self.wi, self.wf, self.wo, self.bc, self.bi, self.bf, self.bo = params

        # states
        self.c = np.zeros(hiddenSize)
        self.i = np.zeros(hiddenSize)
        self.f = np.zeros(hiddenSize)
        self.o = np.zeros(hiddenSize)
        self.s = np.zeros(hiddenSize)
        self.h = np.zeros(hiddenSize)

        self.s_prev = np.zeros_like(self.s)
        self.h_prev = np.zeros_like(self.h)

        self.h_grad = np.zeros_like(self.h)
        self.s_grad = np.zeros_like(self.s)

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
        self.c = np.tanh(np.dot(self.wc, xc) + self.bc)
        self.i = Sigmoid(np.dot(self.wi, xc) + self.bi)
        self.f = Sigmoid(np.dot(self.wf, xc) + self.bf)
        self.o = Sigmoid(np.dot(self.wo, xc) + self.bo)
        self.s = self.c * self.i + self.s_prev * self.f
        self.h = np.tanh(self.s) * self.o

        self.xc = xc

    def Backward(self, prevH_grad, prevS_grad):
        dWo = prevH_grad * np.tanh(self.s) * dSigmoid(self.o)
        dWf = prevS_grad * self.s_prev * dSigmoid(self.f)
        dWi = prevS_grad * self.c * dSigmoid(self.i)
        dWc = self.i * (prevH_grad + prevS_grad) * dTanh(self.c)

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

        self.s_grad = prevS_grad * self.f
        self.h_grad = dxc[self.inputSize:]

        return wc_grad, wi_grad, wf_grad, wo_grad, bc_grad, bi_grad, bf_grad, bo_grad
  
    

def UpdateParams(grads, params, lr=0.1):
    for param, grad in zip(params, grads):
        param -= lr * grad

    return params

def Loss(yLabel, yPred):
    return (yPred[0] - yLabel) ** 2

def Cost(yLabel, yPred):
    return 2 * (yPred[0] - yLabel)


class Lstm():
    def __init__(self, hiddenSize, inputSize):
        self.hiddenSize = hiddenSize
        self.inputSize = inputSize
        
        self.params = InitLstmParams(hiddenSize, inputSize)
        self.cells: List[Lstm_Cell] = []
        self.recurentInputs = [] # inputs


    def Train(self, xList, yList):
        for x in xList:
            self.recurentInputs.append(x)
            # check if there is a node for new input data
            if len(self.recurentInputs) > len(self.cells):
                self.cells.append(Lstm_Cell(self.params, self.hiddenSize, self.inputSize))

            # Forward; if there is no previous cell, then there is no previous cell or hidden state
            idx = len(self.recurentInputs) - 1
            if idx == 0:
                self.cells[idx].Forward(x)
            else:
                # recurent, get previous
                s_prev = self.cells[idx - 1].s
                h_prev = self.cells[idx - 1].h
                self.cells[idx].Forward(x, s_prev, h_prev)

        # Backward
        idx = len(self.recurentInputs) - 1

        loss = Loss(yList[idx], self.cells[idx].h)
        h_grad = Cost(yList[idx], self.cells[idx].h)
        s_grad = np.zeros(self.hiddenSize)

        gradients = self.cells[idx].Backward(h_grad, s_grad)
        self.params = UpdateParams(gradients, self.params)
        idx -= 1

        while idx >= 0:
            loss += Loss(yList[idx], self.cells[idx].h)

            h_grad = Cost(yList[idx], self.cells[idx].h)
            s_grad = self.cells[idx+1].s_grad

            gradients = self.cells[idx].Backward(h_grad, s_grad)
            self.params = UpdateParams(gradients, self.params)
            idx -= 1 

        self.recurentInputs = []
        predictions = [self.cells[i].h[0] for i in range(len(yList))]

        return loss, predictions
  



def Main():
    np.random.seed(0)

    hiddenSize = 100
    inputDim = 50 # inputSize
    lstm = Lstm(hiddenSize, inputDim)
    yLabels = [0.1, 0.2, 0.9, -0.4]
    inputs = [np.random.random(inputDim) for _ in yLabels]

    for i in range(100):
        loss, predictions = lstm.Train(inputs, yLabels)

        print(loss, predictions)
