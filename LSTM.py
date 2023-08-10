import numpy as np


def Sigmoid(x): 
    return 1. / (1 + np.exp(-x))

def dSigmoid(values): 
    return values*(1-values)

def dTanh(values): 
    return 1. - values ** 2


def InitLstmParams(mem_cell_ct, x_dim):
    concat_len = x_dim + mem_cell_ct

    wc = np.random.uniform(-0.1, 0.1, size = (mem_cell_ct, concat_len))
    wi = np.random.uniform(-0.1, 0.1, size = (mem_cell_ct, concat_len))
    wf = np.random.uniform(-0.1, 0.1, size = (mem_cell_ct, concat_len))
    wo = np.random.uniform(-0.1, 0.1, size = (mem_cell_ct, concat_len))

    bc = np.random.uniform(-0.1, 0.1, size = (mem_cell_ct)) 
    bi = np.random.uniform(-0.1, 0.1, size = (mem_cell_ct)) 
    bf = np.random.uniform(-0.1, 0.1, size = (mem_cell_ct)) 
    bo = np.random.uniform(-0.1, 0.1, size = (mem_cell_ct)) 

    return wc, wi, wf, wo, bc, bi, bf, bo


class Lstm_Cell:
    def __init__(self, params, mem_cell_ct, x_dim):       
        self.x_dim = x_dim
        self.mem_cell_ct = mem_cell_ct
        self.wc, self.wi, self.wf, self.wo, self.bc, self.bi, self.bf, self.bo = params

        # states
        self.c = np.zeros(mem_cell_ct)
        self.i = np.zeros(mem_cell_ct)
        self.f = np.zeros(mem_cell_ct)
        self.o = np.zeros(mem_cell_ct)
        self.s = np.zeros(mem_cell_ct)
        self.h = np.zeros(mem_cell_ct)

        self.s_prev = None
        self.h_prev = None

        self.bottom_diff_h = np.zeros_like(self.h)
        self.bottom_diff_s = np.zeros_like(self.s)

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
        self.h = self.s * self.o

        self.xc = xc

    def Backward(self, top_diff_h, top_diff_s):
        dWo = top_diff_h * np.tanh(self.s) * dSigmoid(self.o)
        dWf = top_diff_s * self.s_prev * dSigmoid(self.f)
        dWi = top_diff_s * self.c * dSigmoid(self.i)
        dWc = self.i * (top_diff_h + top_diff_s) * dTanh(self.c)

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

        # save bottom diffs
        self.bottom_diff_s = top_diff_s * self.f # gradient for prev cell state
        #self.bottom_diff_s = ds * self.f # gradient for prev cell state
        self.bottom_diff_h = dxc[self.x_dim:]

        return wc_grad, wi_grad, wf_grad, wo_grad, bc_grad, bi_grad, bf_grad, bo_grad
    

class Lstm:
    pass

class ModelSaver:
    pass


def Main():
    pass
    