def objective_function(task_accuracy, total_cost, agent_count, 
                       accuracy_weight=0.5, cost_weight=0.3, minimality_weight=0.2):
    return (accuracy_weight * task_accuracy) - \
           (cost_weight * total_cost) - \
           (minimality_weight * agent_count)
