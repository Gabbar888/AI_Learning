import numpy as np
import torch
## Don't use this becasue of numerical unstability
def softmax(logits):
    pos_num = np.exp(logits) #(B,C)
    # can be used for both vectors and a batch of vectors using np.array with size (batch_size,num_classes)
    return pos_num/(np.sum(pos_num,axis = -1,keepdims=True)) #(B,C)/(B,1)

def cross_entropy(probs, targets):
    # assume probs is (batch_size, num_classes) and target is (batch_size,) 1d vector
    # ans =0
    # for i in range(target.size):
    #     ans += -np.log(probs[i,target[i]])
    # return ans

    #Clever & Optimised Method using fancy indexing 
    log_of_correct_class = np.log(probs[np.arange(targets.size),targets])
    return -np.mean(log_of_correct_class)


"""
Overflow is a bigger problem than underflow becasue in case of overflow 
the woudl give NaN which is a problem but underflow causes it to
become close to 0 but that is not a problem the class had to be almost 0 probability 
"""

"""
UnderFlow becomes a problem as soon as you deal with log(0)"""

def log_softmax(logits):
    # logits are (batch_size,num_classes)
    m = np.max(logits,axis=-1,keepdims=True)
    # new_logits = logits - (np.ones(logits.shape)*m[:,np.newaxis])
    new_logits = logits - m # Broadcating makes the code easier to write 
    # return np.log(np.exp(new_logits)/(np.sum(np.exp(new_logits),axis = -1)))  ## This is prone to underflow
    log_soft = new_logits - np.log(np.sum(np.exp(new_logits),axis=-1,keepdims=True))
    return log_soft #(B,C)

def softmax_cross_entropy(logits,targets):
    CE_vec = -log_softmax(logits)[np.arange(targets.size),targets]
    return np.mean(CE_vec)

def evaluate(logits,targets):
    ce1 = softmax_cross_entropy(logits,targets)
    logits_torch = torch.tensor(logits,dtype=torch.float32)
    targets_torch = torch.tensor(targets,dtype=torch.long)
    ce2 = torch.nn.functional.cross_entropy(logits_torch,targets_torch)
    print(np.allclose(ce1, ce2.item(), atol=1e-6))

if __name__=="__main__":
    rng = np.random.default_rng(seed=42)
    logits_1 = rng.random(size=(2,4))
    logits_1_ext = np.array([[1000,900,99,999],[81,7,9,57]])
    targets_1 = np.array([1,3])
    logits_2 = rng.random(size=(1,4))
    target_2 = np.array([2])
    evaluate(logits_1,targets_1)
    evaluate(logits_1_ext,targets_1)
    evaluate(logits_2,target_2)
   