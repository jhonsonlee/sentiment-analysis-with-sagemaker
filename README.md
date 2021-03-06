# Deploying a Sentiment Analysis Model
## Project Overview
In this project, I have built a predictive model that receives a movie review from social media and then predicts whether the review contains positive or negative sentiment. If the review is about someone loves the movie then it contains positive sentiment, vice versa. The model was developed in PyTorch framework and deployed using SageMaker on Amazon Web Services. To make it even feels applicable, there is an web app design that receive the review from user and the web app will show the model prediction to the user.

## Setup Instructions

The notebooks provided in this repository are intended to be executed using Amazon's SageMaker platform. The following is a brief set of instructions on setting up a managed notebook instance using SageMaker, from which the notebooks can be completed and run.

### Log in to the AWS console and create a notebook instance

Log in to the AWS console and go to the SageMaker dashboard. Click on 'Create notebook instance'. The notebook name can be anything and using ml.t2.medium is a good idea as it is covered under the free tier. For the role, creating a new role works fine. Using the default options is also okay. Important to note that you need the notebook instance to have access to S3 resources, which it does by default. In particular, any S3 bucket or objectt with sagemaker in the name is available to the notebook.

### Use git to clone the repository into the notebook instance

Once the instance has been started and is accessible, click on 'open' to get the Jupyter notebook main page. We will begin by cloning the SageMaker Deployment github repository into the notebook instance. Note that we want to make sure to clone this into the appropriate directory so that the data will be preserved between sessions.

Click on the 'new' dropdown menu and select 'terminal'. By default, the working directory of the terminal instance is the home directory, however, the Jupyter notebook hub's root directory is under 'SageMaker'. Enter the appropriate directory and clone the repository as follows.

```bash
cd SageMaker
git clone https://github.com/jhonsonlee/sentiment-analysis-with-sagemaker.git
exit
```

After you have finished, close the terminal window.

### Set the AWS Instances

The project needs two computing instaces: `ml.p2.xlarge` and `ml.m4.xlarge`. 
You can view your limit by looking at the EC2 Service Limit report which can be found here: https://console.aws.amazon.com/ec2/v2/home?#Limits.
Your limit for each instace may be `0`. Therefore, you need to submit a request to increase the limit to `1`. Please refer to [Amazon EC2 Service Limits](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-resource-limits.html) for limit management.

### Access the Notebook

Open the `SageMaker Project.ipynb` with Jupyter Notebook and enjoy scripting.

## Warning
When you are not working on this project, make sure that the endpoint are `SHUT DOWN`. You are charged for the length of time that the endpoint is running so if you forget and leave it on you could end up with an unexpectedly large bill.
