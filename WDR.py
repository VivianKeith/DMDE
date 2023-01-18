
"""WDR estimator for evaluate the performance on other policy's historical data."""
import torch
from torch.autograd import Variable

def calc_WDR(
    actions_sequence, rewards_sequence, fence_posts, gamma,
    pi_evaluation, pi_behavior, V = None, Q = None, num_of_actions = None ):

    num_of_trials = len( fence_posts )
    # get weight table
    whole_rho = Variable(torch.zeros((num_of_trials, 21)))
    
    for trial_i in range( num_of_trials ):
        
        rho = 1
        trial_rho = torch.zeros(21)
        trial_rho[0] = rho
        trial_rho = Variable(trial_rho) 
        
        if trial_i < num_of_trials - 1:
            steps_in_trial = fence_posts[ trial_i+1 ] -  fence_posts[ trial_i ]
        else:
            steps_in_trial = actions_sequence.shape[0] - fence_posts[-1]
        for t in range(
                fence_posts[ trial_i], fence_posts[ trial_i ] + steps_in_trial ):
            previous_rho = rho
            rho = rho * (pi_evaluation[ t, actions_sequence[t]] / \
                pi_behavior[ t, actions_sequence[t]])
            trial_aux = torch.zeros(21)
            trial_aux[t - fence_posts[ trial_i] + 1] = 1
            trial_aux = Variable(trial_aux)
            trial_rho = trial_rho + trial_aux * rho
        
        if steps_in_trial < 20:
            for t in range(fence_posts[ trial_i ] + steps_in_trial, fence_posts[ trial_i ] + 20):
                trial_aux = torch.zeros(21)
                trial_aux[t - fence_posts[ trial_i]+1] = 1
                trial_aux = Variable(trial_aux)
                trial_rho = trial_rho + trial_aux * rho
    
        whole_aux = torch.zeros((num_of_trials, 21))
        whole_aux[trial_i, :] = 1
        whole_rho = whole_rho + Variable(whole_aux) * trial_rho
        
    weight_table = whole_rho / torch.sum(whole_rho, dim = 0)
    
    estimator = 0
    # calculate the doubly robust estimator of the policy
    for trial_i in range(num_of_trials):
        discount = 1 / gamma
        if trial_i < num_of_trials - 1:
            steps_in_trial = fence_posts[ trial_i+1 ] -  fence_posts[ trial_i ]
        else:
            steps_in_trial = actions_sequence.shape[0] - fence_posts[-1]
        for t in range(
                fence_posts[ trial_i], fence_posts[ trial_i] + steps_in_trial ):
            previous_weight = weight_table[trial_i, t - fence_posts[ trial_i]]
            weight = weight_table[trial_i, t - fence_posts[ trial_i]+1]
            discount = discount * gamma
            r = rewards_sequence[ t ]
            Q_value = Q[ t, actions_sequence[t]]
            V_value = V[t]
            estimator = estimator + weight * discount * r - discount * ( weight * Q_value - previous_weight * V_value ) 
    
    return estimator