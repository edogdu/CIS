    def extract_timestamp(self, snapshot_id):
        """
        Extracts timestamp from snapshot_id string.
        Format example: 'testbed_system_1_30s_2021-04-19 16:04:30+00:00'
        """
        # Regex to capture YYYY-MM-DD HH:MM:SS
        match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', str(snapshot_id))
        if match:
            return datetime.strptime(match.group(1), '%Y-%m-%d %H:%M:%S')
        # Fallback for integer timestamps or failures
        return datetime.min
