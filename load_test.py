
import sys
import os
import boto3 
import random
import datetime
import math
import time
import numpy as np
     
from concurrent import futures

import sagemaker
from sagemaker import get_execution_role
from sagemaker.serializers import NumpySerializer

def one_thread(endpoint_name, feed_data):
    global latency_list
    global num_infer
    global live
    global num_error
    sagemaker_session = sagemaker.Session()
     
    role = get_execution_role()
    pred = sagemaker.predictor.Predictor(endpoint_name)
    pred.serializer = NumpySerializer()
    
    # Warm up
    for i in range(100):
        output = pred.predict(feed_data)
        feed_data.seek(0)
    time.sleep(3)
    
    # Predictions
    while True:
        start = time.time()
        try:
            pred.predict(feed_data)
        except:
            num_error += 1

        latency = time.time() - start
        latency_list.append(latency*1000/throughput_interval)
        feed_data.seek(0)
        num_infer += batch_size
        if not live:
            break
    
def one_thread_boto3(endpoint_name, feed_data):
    global latency_list
    global num_infer
    global live
    global num_error
     
    client = boto3.client('sagemaker-runtime')
    
    # Warm up
    for i in range(100):
        client.invoke_endpoint(EndpointName=endpoint_name, Body=feed_data)
        feed_data.seek(0)
    time.sleep(3)
    
    # Predictions
    while True:
        start = time.time()
        try:
            client.invoke_endpoint(EndpointName=endpoint_name, Body=feed_data)
        except:
            num_error += 1

        latency = time.time() - start
        latency_list.append(latency*1000/throughput_interval)
        feed_data.seek(0)
        num_infer += batch_size
        if not live:
            break
    
     
def current_performance():
    last_num_infer = num_infer
    print(' TPS  |  P50  |  P90  |  P95  |  P99  |  err  ')
     
    for _ in range(throughput_time // throughput_interval):
     
        current_num_infer = num_infer
        throughput = (current_num_infer - last_num_infer) / throughput_interval
        client_avg = 0.0
        client_p50 = 0.0
        client_p90 = 0.0
        client_p95 = 0.0
        client_p99 = 0.0
        if latency_list:
            client_avg = np.mean(latency_list[-latency_window_size:])
            client_p50 = np.percentile(latency_list[-latency_window_size:], 50)
            client_p90 = np.percentile(latency_list[-latency_window_size:], 90)
            client_p95 = np.percentile(latency_list[-latency_window_size:], 95)
            client_p99 = np.percentile(latency_list[-latency_window_size:], 99)
        print('{:5.3f}|{:.5f}|{:.5f}|{:.5f}|{:.5f} |{:4d}'.format(throughput, client_p50, client_p90, client_p95, client_p99, int(num_error) ))
        last_num_infer = current_num_infer
     
        time.sleep(throughput_interval)
    global live
    live = False
    
def check_endpoint_exists(endpoint_name):
    try:
        client = boto3.client("sagemaker")
        status = client.describe_endpoint(EndpointName=endpoint_name)['EndpointStatus']
        if status == 'InService':
            return True
        else:
            raise
    except:
        return False
    
def load_tester(num_thread, endpoint_name, filename, request_type):
    
    global throughput_interval
    throughput_interval = 10
    global throughput_time
    throughput_time = 200
    global latency_window_size
    latency_window_size = 1000
    global batch_size
    batch_size = 1
    global live
    live = True
    global num_infer
    num_infer = 0
    global latency_list
    latency_list = []
    global num_error
    num_error = 0
    
    try:
        assert check_endpoint_exists(endpoint_name)
    except AssertionError:
        print(f'The endpoint {endpoint_name} does not exist or is not available.')
    else:
        if request_type == 'sm':
            print('Using SageMaker Python SDK for requests.')
        else:
            print('Using boto3 for requests.')
        executor = futures.ThreadPoolExecutor(max_workers=num_thread+1)
        executor.submit(current_performance)
        if request_type == 'sm':
            for pred in range(num_thread):
                executor.submit(one_thread, endpoint_name, open(filename, 'rb'))
        else:
            for pred in range(num_thread):
                executor.submit(one_thread_boto3, endpoint_name, open(filename, 'rb'))
        executor.shutdown(wait=True)

        
if __name__ == '__main__':
        
    num_thread = int(sys.argv[1]) # First cmd line argument: number of concurrent client threads (int)
    endpoint_name = sys.argv[2] # Second command line argument: SageMaker Endpoint Name (str)
    filename = sys.argv[3] # Name of the input image file
    request_type = sys.argv[4] #boto3 or SM python sdk 

    load_tester(num_thread, endpoint_name, filename, request_type)