import json
import boto3
import os

def lambda_handler(event, context):
    # Initialize Bedrock client
    bedrock = boto3.client('bedrock-runtime')
    
    # Handle CORS preflight requests
    if event.get('httpMethod') == 'OPTIONS':
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Methods': 'POST,OPTIONS'
            },
            'body': ''
        }
    
    try:
        # Parse the request body
        body = json.loads(event.get('body', '{}'))
        
        # Extract context and question from the request body
        pdf_text = body.get('context', '')
        question = body.get('question', '')
        
        if not pdf_text or not question:
            return {
                'statusCode': 400,
                'headers': {
                    'Access-Control-Allow-Origin': '*',
                    'Content-Type': 'application/json'
                },
                'body': json.dumps({'error': 'Missing context or question'})
            }
        
        # Construct the prompt
        prompt_data = f"""Answer the question based only on the information provided between ##.
Only answer the question if you can find relevant information in the context, otherwise, answer "I don't have enough information to answer this question".
#
{pdf_text}
#

Question: {question}
Answer:"""
        
        # Bedrock parameters
        parameters = {
            "maxTokenCount": 512,
            "stopSequences": [],
            "temperature": 0,
            "topP": 0.9
        }
        
        # Prepare the request body
        bedrock_body = json.dumps({
            "inputText": prompt_data,
            "textGenerationConfig": parameters
        })
        
        # Invoke Bedrock model
        response = bedrock.invoke_model(
            body=bedrock_body,
            modelId="amazon.titan-tg1-large",
            accept="application/json",
            contentType="application/json"
        )
        
        # Parse the response
        response_body = json.loads(response.get("body").read())
        answer = response_body.get("results")[0].get("outputText")
        
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Content-Type': 'application/json'
            },
            'body': json.dumps({
                'answer': answer.strip()
            })
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Content-Type': 'application/json'
            },
            'body': json.dumps({
                'error': str(e)
            })
        } 