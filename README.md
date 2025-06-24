# Continual-learning-for-very-short-term-load-forecasting-a-case-study-on-parts-cleaning

## Summary

This repository contains the supplementary materials for the paper:

**"Continual Learning for Very Short-Term Load Forecasting: A Case
Study on Parts Cleaning"**  
by Robin Zink, Jonathan Magin, Oliver Griess, Matthias Weigold 
Submitted to <span style="color:red">*[Journal Name]*, [Year]</span>.  
Available at: <span style="color:red">DOI</span>

The paper investigates continual learning for industrial load forecasting, and proposes a framework including performance monitoring. Our approach is evaluated on a [dataset](https://tudatalib.ulb.tu-darmstadt.de/items/ec2f93e4-8994-4834-be60-351b8dc04c6d) of a industrial throughput parts cleaning machine, showing increased robustness of load forecasting model operations in industrial energy systems.

## Repository structure

```
[project_root/](https://github.com/r-squared-ptw/Continual-learning-for-very-short-term-load-forecasting-a-case-study-on-parts-cleaning/tree/)
├── main/ # Main code folder
│ ├── EWC.py # elastic weight consolidation
│ ├── LWF.py # learning without forgetting
│ ├── MAS.py # memory aware synapses
│ ├── OEWC.py # online elastic weight consolidation
│ ├── SI.py # synaptic intelligence
│ ├── __init__.py # initialization
│ ├── cdd_regression.py # concept drift detection for regression
│ └── my_monitoring_metrics.py # continual learning performance monitoring
└── ...
```

## Licence

The project uses the [MIT License](https://github.com/r-squared-ptw/Continual-learning-for-very-short-term-load-forecasting-a-case-study-on-parts-cleaning/blob/main/LICENSE), so feel free to use it for your own projects.

## Contributors

- Robin Zink
- Oliver Griess