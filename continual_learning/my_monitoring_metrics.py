import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error

def compute_backward_transfer(models, tasks_data, postprocessing = None):
    """
    Compute Backward Transfer (BWT) over multiple tasks.

    This metric quantifies how learning new tasks affects performance on previously 
    learned tasks. Specifically, it measures the change in error on past tasks before 
    and after learning subsequent tasks. A positive BWT value indicates that later 
    tasks improve earlier task performance (constructive interference), while a 
    negative value indicates forgetting (interference).

    The backward transfer value corresponds to the average decrease (positive BWT) 
    or increase (negative BWT) in MAE on past tasks as training progresses.

    Parameters:
        models (list): List of trained models, where each model corresponds to a task.
        tasks_data (list of tuples): List of (X, y) pairs for each task, where:
            - X (numpy array): Input features for the task.
            - y (numpy array): Corresponding target values.
        postprocessing (callable, optional): An optional transformation to apply
            to model predictions (e.g., inverse scaling). Must have `.inverse_transform()`.

    Returns:
        float: Backward transfer value. Positive values indicate beneficial backward transfer,
               negative values indicate forgetting.

    Raises:
        ValueError: If fewer than two tasks are provided (n_tasks < 2).

    Reference (adapted for regression tasks):
    Lopez-Paz, D., & Ranzato, M. (2017). Gradient Episodic Memory for Continual Learning.
    https://arxiv.org/pdf/1706.08840
    """
    n_tasks = len(tasks_data)
    if n_tasks < 2:
        raise ValueError("At least two tasks are required to compute backward transfer.")
    test_losses = np.zeros((n_tasks, n_tasks))

    for i in range(n_tasks):
        for j in range(n_tasks):
            X_task_j, y_task_j = tasks_data[j]

            # Evaluate model performance on task j
            y_pred_task_j = models[i].predict(X_task_j)
            if postprocessing is not None:
                y_pred_task_j = postprocessing.inverse_transform(y_pred_task_j)
            error_task_j = mean_absolute_error(y_task_j, y_pred_task_j)

            # Record the test MSE for task j after observing the last sample from task i
            test_losses[i, j] = error_task_j

    # Compute backward transfer
    sum_backward_transfer = 0
    for i in range(1, n_tasks):
        sum_backward_transfer += np.mean(test_losses[i-1, :i] - test_losses[i, :i])

    backward_transfer = sum_backward_transfer / (n_tasks - 1)
    
    return backward_transfer


def compute_forward_transfer(models, tasks_data, postprocessing = None):
    """
    Compute Forward Transfer (FWT) across multiple tasks.

    This metric quantifies how learning previous tasks influences performance on 
    future, yet-to-be-learned tasks. Specifically, it compares model performance 
    on future tasks before and after those tasks are trained. A positive FWT value 
    indicates that earlier learning benefits future performance (positive transfer), 
    while a negative value suggests interference.

    The forward transfer value corresponds to the average decrease (positive FWT) 
    or increase (negative FWT) in MAE on future tasks.

    Parameters:
        models (list): List of trained models, where each model corresponds to a task.
        tasks_data (list of tuples): List of (X, y) pairs for each task, where:
            - X (numpy array): Input features for the task.
            - y (numpy array): Corresponding target values.
        postprocessing (callable, optional): An optional transformation applied to 
            model predictions (e.g., inverse scaling). Must implement `.inverse_transform()`.

    Returns:
        float: Forward transfer value. Higher values indicate stronger positive transfer.

    Raises:
        ValueError: If fewer than three tasks are provided (n_tasks < 3).

    Reference (adapted for regression tasks):
        Lopez-Paz, D., & Ranzato, M. (2017). Gradient Episodic Memory for Continual Learning.
        https://arxiv.org/pdf/1706.08840
    """
    n_tasks = len(tasks_data)
    if n_tasks < 3:
        raise ValueError("At least three tasks are required to compute forward transfer.")
    test_losses = np.zeros((n_tasks, n_tasks))

    # Evaluate models on all tasks
    for i in range(n_tasks):
        for j in range(n_tasks):
            X_task_j, y_task_j = tasks_data[j]

            # Evaluate model performance on task j
            y_pred_task_j = models[i].predict(X_task_j)
            if postprocessing is not None:
                y_pred_task_j = postprocessing.inverse_transform(y_pred_task_j)
            error_task_j = mean_absolute_error(y_task_j, y_pred_task_j)

            # Record the test loss for task j after observing the last sample from task i
            test_losses[i, j] = error_task_j

    # Compute forward transfer
    sum_forward_transfer = 0
    for i in range(1, n_tasks-1):
        sum_forward_transfer +=  np.mean(test_losses[i-1, i+1:] - test_losses[i, i+1:])

    forward_transfer = sum_forward_transfer / (n_tasks - 2)

    return forward_transfer


def compute_model_size_efficiency(models):
    """
    Compute model size efficiency across a sequence of models.

    This metric evaluates how efficiently model parameters are reused across tasks
    by comparing the memory footprint of each task-specific model to that of the 
    first model. A higher value indicates better parameter efficiency.

    Parameters:
    models (list of models): List containing a trained model for each task.
                             Each model must implement the `count_params()` method,
                             which returns the total number of trainable parameters.

    Returns:
    model_size_efficiency (float): Model size efficiency value in [0, 1]. Higher is better.

    Raises:
    ValueError: If fewer than two models are provided.

    Reference:
    Lopez-Paz, D., & Ranzato, M. (2017). Gradient Episodic Memory for Continual Learning.
    https://arxiv.org/pdf/1810.13166
    """
    
    def get_model_memory_size(model):
        num_params = model.count_params()
        memory_size = num_params * 4  # assuming each parameter is a float32 which is 4 bytes
        return memory_size
    
    if len(models) < 2:
        raise ValueError("At least two tasks are required to compute model size efficiency.")

    mem_theta1 = get_model_memory_size(models[0])
    N = len(models)
    mem_thetas = [get_model_memory_size(model) for model in models]
    
    sum_mem_ratios = sum(mem_theta1 / mem_theta for mem_theta in mem_thetas)
    model_size_efficiency = min(1, sum_mem_ratios / N)
    
    return model_size_efficiency

def compute_class_size_efficiency(class_sizes):
    """
    Compute class size efficiency based on a list of class sizes per task.

    This metric assesses the development of the class size across tasks by comparing 
    the class size of the first task to the sizes of all tasks. A higher value 
    indicates better efficiency in managing class size.

    Parameters:
    class_sizes (list of int or float): List containing the class size per task.

    Returns:
    model_size_efficiency (float): Class size efficiency value in [0, 1]. Higher is better.

    Raises:
    ValueError: If fewer than two class sizes are provided.
    """

    if len(class_sizes) < 2:
        raise ValueError("At least two tasks are required to compute class size efficiency.")
    
    mem_theta1 = class_sizes[0]
    N = len(class_sizes)
    mem_thetas = class_sizes
    
    sum_mem_ratios = sum(mem_theta1 / mem_theta for mem_theta in mem_thetas)
    model_size_efficiency = min(1, sum_mem_ratios / N)
    
    return model_size_efficiency

def compute_sss_efficiency(samples, all_examples):
    """
    Compute the Samples Storage Size (SSS) efficiency.

    This metric evaluates how efficiently a method stores previously seen samples 
    compared to the total memory footprint of all observed data, normalized by 
    the number of tasks. A higher value indicates more efficient use of memory.

    Parameters:
        samples (list of numpy arrays): List of stored samples per task.
        all_examples (list of numpy arrays): List of all examples encountered across tasks.
        N (int): Number of tasks.

    Returns:
        float: SSS efficiency value in [0, 1]. Higher is better.

    Raises:
        ValueError: If the number of tasks N is less than 1.

    Reference:
    Lopez-Paz, D., & Ranzato, M. (2017). Gradient Episodic Memory for Continual Learning.
    https://arxiv.org/pdf/1810.13166
    """
    def calculate_memory_occupation(samples):
        """
        Calculate the total memory occupation of a list of samples.
        
        Args:
            samples (list of numpy arrays): List of samples.
        
        Returns:
            int: Total memory occupation in bytes.
        """
        return sum(sample.nbytes for sample in samples)
    
    if len(samples) < 1:
        raise ValueError("At least one task is required to compute sample storage size efficiency.")

    Mem_D = calculate_memory_occupation(all_examples)
    
    mem_ratios = []
    for task_samples in samples:
        Mem_Mi = calculate_memory_occupation(task_samples)
        mem_ratios.append(Mem_Mi / Mem_D)

    SSS = 1 - min(1, sum(mem_ratios) / len(samples))
    return SSS

def calculate_ce_score(train_times):
    """
    Compute the Computational Efficiency (CE) score.

    This metric evaluates how efficiently training time is managed across tasks 
    by comparing the training time of the first model to all subsequent ones. 
    A higher CE score indicates better computational efficiency in terms of 
    training duration.

    Parameters:
        train_times (list of float): List containing training times (e.g., in seconds)
                                     for each task-specific model.

    Returns:
        float: CE score value in [0, 1]. Higher is better.

    Raises:
        ValueError: If fewer than two training times are provided.

    Reference, but adapted to use training times:
    Lopez-Paz, D., & Ranzato, M. (2017). Gradient Episodic Memory for Continual Learning.
    https://arxiv.org/pdf/1810.13166
    """

    if len(train_times) < 2:
        raise ValueError("At least two tasks are required to compute CE score.")
    
    mem_theta1 = train_times[0]
    N = len(train_times)
    mem_thetas = train_times
    
    sum_mem_ratios = sum(mem_theta1 / mem_theta for mem_theta in mem_thetas)
    ce_score = min(1, sum_mem_ratios / N)
    
    return ce_score