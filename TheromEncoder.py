

import numpy as np
import matplotlib.pyplot as plt
import math

def count(arr):
	tally = dict()
	for ele in arr:
		if ele not in tally.keys():
			tally[ele] = 0
		tally[ele] += 1
	return tally


def save_plot(x, y, name):
	plt.plot(x, y, 'o-')
	plt.xlim((0,100))
	plt.savefig(name, bbox_inches="tight")
	plt.cla()





# default is same frequent interval
def DefaultTheromEncoder(x, y, encoding_len=8):
	temp_data = np.reshape(np.concatenate((X_train, X_valid, X_test)),(-1))
	temp_data.sort()
	print(temp_data)
	tally = count(temp_data)
	print(tally)

	# save_plot(temp_data.keys(), temp_data.values(), "images/test.jpg")

	total_num = len(temp_data)
	print('total_num: ', str(total_num))
	each_part = total_num / encoding_len
	print('each part num: ', str(each_part))
	each_part_int = int(each_part)
	print('each part int num: ', str(each_part_int))

	splited_data = []
	start_idx = 0
	for i in range(encoding_len):
		end_idx = start_idx + each_part_int
		print('start idx: ', str(start_idx))
		print('end idx: ', str(end_idx))


		# if next element is same as last one in sub arr, move it into current sub array
		while end_idx < total_num and temp_data[end_idx-1] == temp_data[end_idx]:
			end_idx+=1

		cur_sub_arr = temp_data[start_idx:end_idx]
		splited_data.append(cur_sub_arr)

		start_idx = end_idx

	print(splited_data)


# handle zero individually 
def SpecialTheromEncoder(x, y, encoding_len=8, debug_mode=False):
	temp_data = np.reshape(np.concatenate((X_train, X_valid, X_test)),(-1))
	temp_data.sort()
	if debug_mode:
		print(temp_data)
	tally = count(temp_data)
	if debug_mode:
		print(tally)

	# save_plot(temp_data.keys(), temp_data.values(), "images/test.jpg")


	temp_data_without_zero = temp_data[tally[0.0]:]
	total_num = len(temp_data_without_zero) 
	if debug_mode:
		print('total_num: ', str(total_num))
	each_part = total_num / encoding_len
	if debug_mode:
		print('each part num: ', str(each_part))
	each_part_int = int(each_part)
	if debug_mode:
		print('each part int num: ', str(each_part_int))

	splited_data = []
	start_idx = 0
	for i in range(encoding_len):
		end_idx = start_idx + each_part_int
		if debug_mode:
			print('start idx: ', str(start_idx))
			print('end idx: ', str(end_idx))


		# if next element is same as last one in sub arr, move it into current sub array
		while end_idx < total_num and temp_data_without_zero[end_idx-1] == temp_data_without_zero[end_idx]:
			end_idx+=1

		cur_sub_arr = temp_data_without_zero[start_idx:end_idx]
		splited_data.append(cur_sub_arr)

		start_idx = end_idx

	if debug_mode:
		print(splited_data)

		for i in splited_data:
			print(i)
			print(len(i))






class CustomizedTheromEncoder():
	def __init__(self, encoding_len=8, debug_mode=False):
		self.thresholds = []
		self.debug_mode=debug_mode
		self.encoding_len = encoding_len

	def fit(self, data):
		temp_data = data
		temp_data.sort()
		if self.debug_mode:
			print(temp_data)
		tally = count(temp_data)
		if self.debug_mode:
			print(tally)

		temp_data_without_zero = temp_data[tally[0.0]:]
		total_num = len(temp_data_without_zero) 
		if self.debug_mode:
			print('total_num: ', str(total_num))
		each_part = total_num / self.encoding_len
		if self.debug_mode:
			print('each part num: ', str(each_part))
		each_part_int = int(each_part)
		if self.debug_mode:
			print('each part int num: ', str(each_part_int))

		splited_data = []
		start_idx = 0
		for i in range(self.encoding_len):
			end_idx = start_idx + each_part_int
			if self.debug_mode:
				print('start idx: ', str(start_idx))
				print('end idx: ', str(end_idx))


			# if next element is same as last one in sub arr, move it into current sub array
			while end_idx < total_num and temp_data_without_zero[end_idx-1] == temp_data_without_zero[end_idx]:
				end_idx+=1

			cur_sub_arr = temp_data_without_zero[start_idx:end_idx]
			splited_data.append(cur_sub_arr)

			start_idx = end_idx

		if self.debug_mode:
			print(splited_data)

			for i in splited_data:
				print(i)
				print(len(i))


		for sub in splited_data:
			if len(sub) != 0:
				self.thresholds.append(sub[0])

		if self.debug_mode:
			print("thresholds")
			print(self.thresholds)

	
	def encode(self, val):
		result = []
		for t in self.thresholds:
			if val > t:
				result.append(1)
			else:
				result.append(0)
		return result


	def transform(self, data):
		result = []
		for row in data:
			new_row = []
			for ele in row:
				new_row.append(self.encode(ele))
			result.append(new_row)

		return result


class StandardTheromEncoder:
    def __init__(self, encoding_len=8,debug_mode=False):
        self.interval = 0
        self.debug_mode = debug_mode
        self.encoding_len = encoding_len

    def fit(self, data):
        max_num = np.max(data)
        min_num = np.min(data)
        if self.debug_mode:
            print('max num: ', str(max_num))
            print('min num: ', str(min_num))

        self.interval = math.ceil((max_num - min_num) / self.encoding_len)
        if self.debug_mode:
            print('interval: ', str(self.interval))

    def encode(self, val):
        result = []
        for i in range(self.encoding_len):
            if val > i * self.interval:
                result.append(1)
            else:
                result.append(0)
        return result

    def transform(self, data):
        result = []
        for row in data:
            new_row = []
            for ele in row:
                new_row.append(self.encode(ele))
            result.append(new_row)

        return result
