
import json
import boto3

runtime_sm_client = boto3.client('sagemaker-runtime')

def lambda_handler(event, context):
    try:
        video_path = event['Records'][0]['s3']['object']['key']
        video_name = video_path.split('/')[-1]
        output_path = "output/output_" + video_name

        content_type = "application/json"
        request_body = {"input_path": video_path, "output_path": output_path}
        payload = json.dumps(request_body)

        endpoint_name = 'yolov8-handwave-endpoint'

        response = runtime_sm_client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType=content_type,
            Body=payload
        )

        response_body = response['Body'].read().decode('utf-8')
        parsed_response = json.loads(response_body)

        output_data = parsed_response.get('output')

        print(output_data)

        return {
            'statusCode': 200,
            'body': json.dumps('SageMaker endpoint invoked successfully.')
        }

    except Exception as e:
        print(f"Error: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps('Error invoking SageMaker endpoint.')
        }
