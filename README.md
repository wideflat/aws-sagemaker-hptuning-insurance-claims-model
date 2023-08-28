# AWS SageMaker Hyperparameter Tuning for Insurance Claims Model

## About
One of the benefits of using AWS SageMaker for machine learning is that it allows us to distribute hyperparameter tuning tasks into multiple instances.
SageMaker has a training API called [HyperparameterTuner](https://sagemaker.readthedocs.io/en/stable/api/training/tuner.html). However, I barely found any comprehensive examples.
And what's more difficult is that most of the AWS SageMaker documentations are using XGBoost as an example.

As a pricing actuary/data scientist working in the insurance industry, I wanted to use Scikit-learn [TweedieRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TweedieRegressor.html), which is obviously not a popular modeling algorithm, and I barely found any examples online.

So I explored SageMaker's official documentations and StackOverflow posts, and finally figured out how to run hyperparameter tuning in SageMaker with Scikit-learn TweedieRegresson for insurance claims model .

Hope you will find this repository helpful:)


## Data
I'm using open-source dataset, French Motor Third Party Claims. The official documentation is in here: http://cas.uqam.ca/pub/web/CASdatasets-manual.pdf.


## Process

To train a Scikit-learn model and hyperparameter tuning by using the SageMaker Python SDK:
1. Prepare a training script
2. Create a sagemaker.sklearn.SKLearn Estimator
3. Create a sagemaker.tuner.Hyperparameter Tuner
4. Call the tuner's fit method

![alt text](https://github.com/wideflat/aws-sagemaker-hptuning-insurance-claims-model/blob/main/images/flow.png)


### 1. Prepare a training script
- This is a typical training script you use outside of SageMaker and it will be passed to sagemaker.sklearn.SKLearn Estimator later on.
- Hyperparameters that you will tune should be passed to your script as arguments. In this repository, I use Scikit-learn's TweedieRegressor and two hyperparameters `alpha` and `power` are parsed in arguments.

```bash
    parser = argparse.ArgumentParser()

    parser.add_argument("--nfolds", type=int, default=3)
    parser.add_argument("--scoring", type=str, default='neg_root_mean_squared_error')
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--power", type=float, default=1.5)
    
    # SageMaker specific arguments. Defaults are set in the environment variables
    # Location of input training data
    parser.add_argument("--train_data_dir", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    # Location of input validation data
    parser.add_argument("--validation_data_dir", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION"))
    # Location where trained model will be stored. Default set by SageMaker, /opt/ml/model
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    # Location where model artifacts will be stored. Default set by SageMaker, /opt/ml/output/data
    parser.add_argument("--output_data_dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR"))
    
    args = parser.parse_args()
```

Another important point to make HyperparameterTuner work is to print out a metric used to select the best performance model in hyperparmeter tuning.
In this case, I use `cv_rmse` as a metric to select the best performance model and I print it out with `print(f"[0]#011cv-rmse:{cv_rmse:.2f}")`. This will be identified as a flag by HyperparameterTuner in Python SDK.

```bash
    print(f"[0]#011train-rmse:{rmse_train:.2f}")
    print(f"[0]#011validation-rmse:{rmse_valid:.2f}")
    print(f"[0]#011cv-rmse:{cv_rmse:.2f}")

    metrics_data = {
                    "metrics": {
                                "train:rmse": {"value": rmse_train},
                                "validation:rmse": {"value": rmse_valid},
                                "cv:rmse": {"value": cv_rmse},
                                }
                   }
```


### 2. Create a sagemaker.sklearn.SKLearn Estimator

You run Scikit-learn training scripts on SageMaker by creating SKLearn Estimators.

```bash
sklearn = SKLearn(
    entry_point="sktweedie_train.py",
    framework_version="1.2-1",
    instance_type="ml.m5.xlarge",
    instance_count=1,
    role=sagemaker_role,
    sagemaker_session=sagemaker_session,
    # hyperparameters=static_hyperparams,
    output_path=output_path,
    # code_location=estimator_output_uri,
    base_job_name="frenchtpl-sktweedie-hptune"
)
```

### 3. Create a sagemaker.tuner.Hyperparameter Tuner
You configure ranges of hyperparameters and passed it onto `HyperparameterTuner`.

```bash
hyperparameter_ranges = {
    "alpha": ContinuousParameter(0, 1),
    "power": ContinuousParameter(1.5, 1.9)
}

objective_metric_name = "cv-rmse"
metric_definitions = [{'Name': 'cv-rmse',
                       'Regex': '.*\[[0-9]+\].*#011cv-rmse:([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?).*'}]

tuner = HyperparameterTuner(
    estimator = sklearn,
    objective_metric_name = objective_metric_name,
    metric_definitions = metric_definitions,
    hyperparameter_ranges = hyperparameter_ranges,
    base_tuning_job_name = "frenchtpl-sktweedie-hptune",
    max_jobs=5,
    max_parallel_jobs=2,
    objective_type = 'Minimize',
    strategy = "Random",
)
```

### 4. Call the tuner's fit method
Finally, you configure train and validation dataset, and call the tuner's fit method.

```bash
# Setting the input channels for tuning job
s3_input_train = TrainingInput(s3_data=train_data_uri, content_type="csv", s3_data_type="S3Prefix")
s3_input_validation = TrainingInput(s3_data=validation_data_uri, content_type="csv", s3_data_type="S3Prefix")

tuner.fit(inputs={"train": s3_input_train, "validation": s3_input_validation})
tuner.wait()
```

While hyperparameter tuning job is in progress, you will see the status of the progress as below in the notebook.
![alt text](https://github.com/wideflat/aws-sagemaker-hptuning-insurance-claims-model/blob/main/images/image1.png)

Once the hyperparameter tuning job is completed, all the individual models and their metrics are displayed in your SageMaker Experiments page as below.
![alt text](https://github.com/wideflat/aws-sagemaker-hptuning-insurance-claims-model/blob/main/images/image2.png)


## Reference

Using Scikit-learn with the SageMaker Python SDK
https://sagemaker.readthedocs.io/en/stable/frameworks/sklearn/using_sklearn.html

SageMaker HyperparameterTuner
https://sagemaker.readthedocs.io/en/stable/api/training/tuner.html


### ...END
