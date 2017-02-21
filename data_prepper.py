import csv

def get_and_training():
	data = []
	with open('data/training_and.csv', 'rb') as csvfile:
		rows = csv.reader(csvfile, delimiter=',')
		for row in rows:
			inp = [int(row[0]), int(row[1])]
			op = [int(row[2])]
			data.append((inp, op))
	return data

def get_or_training():
	data = []
	with open('data/training_or.csv', 'rb') as csvfile:
		rows = csv.reader(csvfile, delimiter=',')
		for row in rows:
			inp = [int(row[0]), int(row[1])]
			op = [int(row[2])]
			data.append((inp, op))
	return data

def get_xor_training():
	data = []
	with open('data/training_xor.csv', 'rb') as csvfile:
		rows = csv.reader(csvfile, delimiter=',')
		for row in rows:
			inp = [int(row[0]), int(row[1])]
			op = [int(row[2])]
			data.append((inp, op))
	return data
