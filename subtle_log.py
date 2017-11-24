import os
import datetime

class Logger(object):
	""" keep track of log and time """
	def __init__(self, name="subtle_logger"):
		self.name = name
		self.list_log = []
		self.list_event = []
		self.time_init = datetime.datetime.now()

	def update_event(self, event_name):
		event = {
			"name": event_name,
			"time": datetime.datetime.now(),			
		}
		event['duration'] = event["time"] - self.time_init
		self.list_event.append(event)
		return event

	def get_events(self):
		return self.list_event

	def get_logs(self):
		return self.list_log

	def log_event(self, event_name, event_description=""):
		event = self.update_event(event_name)
		event['descritpion'] = event_description
		print('Subtle Log: event {0}: time {1}, {2}'.format(
			event_name, 
			event['duration'],
			event_description
			))
