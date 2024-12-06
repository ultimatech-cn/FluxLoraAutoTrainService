import pandas as pd
from datetime import datetime
import os
from job_status import JobStatus
import base64
from logger_config import setup_logger
from common_tools import project_config
logger = setup_logger('job_record_tools')

class JobStatusManager:
    def __init__(self, csv_file='job_status.csv'):
        self.csv_file = csv_file
        if not os.path.exists(csv_file):
            # Create new file with specified column order
            df = pd.DataFrame({
                'image_path': [],  # Put image_path at the front
                'jobid': [],
                'job_type': [],
                'status': [],
                'completion_time': [],
                'caption': [],
                'model_name': []
            })
            df = df.astype({
                'image_path': str,  # Ensure column order matches above
                'jobid': str,
                'job_type': str,
                'status': str,
                'completion_time': str,
                'caption': str,
                'model_name': str
            })
            df.to_csv(csv_file, index=False)
    
    def add_job(self, jobid, image_path, job_type, caption, model_name, status=JobStatus.Processing.value):
        """Add new workflow record"""
        df = pd.read_csv(self.csv_file)
        
        # Add new record, adjust field order
        new_record = pd.DataFrame([{
            'image_path': image_path,
            'jobid': jobid,
            'job_type': job_type,
            'status': status,
            'completion_time': None,
            'caption': caption,
            'model_name': model_name
        }])
        df = pd.concat([new_record, df], ignore_index=True)
        print(f"df: {df}")
        df.to_csv(self.csv_file, index=False)
    
    def update_job_status(self, jobid, job_type, new_status):
        """Update workflow status"""
        df = pd.read_csv(self.csv_file)
        
        # Ensure correct type for all columns
        df = df.astype({
            'image_path': str,
            'jobid': str,
            'job_type': str,
            'status': str,
            'completion_time': str,
            'caption': str,
            'model_name': str
        })
        
        # Check if jobid exists
        if jobid not in df['jobid'].values:
            raise ValueError(f'JobID {jobid} does not exist')
        
        # Update status
        mask = (df['jobid'] == jobid & df['job_type'] == job_type)
        df.loc[mask, 'status'] = new_status
        
        # If status is done, update completion time
        if new_status.lower() == JobStatus.Done.value or new_status.lower() == JobStatus.Failed.value:
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            df.loc[mask, 'completion_time'] = current_time
        
        print(f"df: {df}")
        df.to_csv(self.csv_file, index=False)

    def get_all_records(self):
        """
        Returns:
            pandas.DataFrame: DataFrame containing all records
        """
        try:
            if os.path.exists(self.csv_file):
                df = pd.read_csv(self.csv_file)
                print(df.head())
                records = df.head(10).values.tolist()
                for record in records:
                    img_path = record[0]
                    if os.path.exists(img_path):
                        # Read image and convert to base64
                        with open(img_path, "rb") as img_file:
                            img_data = base64.b64encode(img_file.read()).decode('utf-8')
                        # Modify style attribute to display image as square
                        img_type = img_path.split('.')[-1]
                        record[0] = f'<img src="data:image/{img_type};base64,{img_data}" style="width: 75px; height: 75px; object-fit: cover;">'
                    else:
                        record[0] = "Image does not exist"
                return records
            return [["No active tasks", "", ""]]
        except Exception as e:
            return [["Error reading task status", str(e), ""]]
        
    def get_pending_jobs(self):
        """
        Get tasks that are waiting or in processing status, and convert to JSON format
        Returns:
            list: List of dictionaries containing task data
        """
        try:
            if not os.path.exists(self.csv_file):
                return []
                
            df = pd.read_csv(self.csv_file)
            # Get first queue_size + 1 rows
            df = df.head(project_config['queue_size'] + 1)
            
            # Filter records with WaitingQueue or Processing status
            mask = (df['status'] == JobStatus.WaitingQueue.value) | (df['status'] == JobStatus.Processing.value)
            pending_df = df[mask]
            
            # Convert to list of task data, iterate from back to front
            pending_jobs = []
            for idx in range(len(pending_df)-1, -1, -1):
                row = pending_df.iloc[idx]
                task_data = {
                    "job_id": row['jobid'],
                    "job_type": row['job_type'],
                    "model_name": row['model_name'],
                    "image_path": row['image_path'],
                    "caption": row['caption']
                }
                pending_jobs.append(task_data)
                
            return pending_jobs
            
        except Exception as e:
            logger.error(f"Error getting pending jobs: {str(e)}")
            return []
    
    
    