# only extracts features, labels are outside the scope of this code

from scipy.io import loadmat
import numpy
import math
frame_times = [365, 456, 7200] # pls fill in Yipeng in seconds

mini_window = 0.0200
overlap = 0.0100
ahead = 0.0990
back = 0.0200

def ret_vec_per_neuron(s, e, neuron, data):
	
	actual_start = [s - mini_window + overlap]
	temp = s
	while temp < e:
		actual_start.append(temp)
		temp = temp + mini_window - overlap

	actual_end = [start_temp + mini_window for start_temp in actual_start]
	vec_output = [0]*len(actual_start)

	for k ,(start, end) in enumerate(zip(actual_start, actual_end)):
		for d in data['arr']:
			if float(d) >= start and float(d) <= end:
				vec_output[k] += 1
	return vec_output





def simpler_features_yip():
	start_range = []
	end_range = []

	time_start = 0.0000
	time_end = 1.0000

	vec_output = []

	while time_start < 42.00*60.00:
		sub = numpy.linspace(time_start, time_end, num=15, endpoint=False)
		start_range.extend(sub)
		end_range.extend([s + 1.0000/15.0000 for s in sub])

		time_start = time_end
		time_end = time_end + 1

	for neuron in range(1,139):
		vec = [0]*len(start_range)

		try:
			data = loadmat('Fried_Data/Neuron' + str(neuron) + '.mat')
			for d in data['arr']:
				for k ,(start, end) in enumerate(zip(start_range, end_range)):
					if float(d[0]) >= start and float(d[0]) < end:
						vec[k] += 1
		except:
			pass

		print(sum(vec))
		vec_output.append(vec)

	import pickle
	with open('features_mat138x15x60x48.pkl', 'wb') as f:
		pickle.dump(vec_output, f)


def basic_features():
	vec_output = simpler_features_yip()


def get_features():
	for frame in frame_times:
		start_time = frame - back
		end_time = frame + ahead
		frame_vec = []
		for neuron in range(1,139):
			flag = 0
			try:
				data = loadmat('Fried_Data/Neuron' + str(neuron) + '.mat')
				flag = 1
			except:
				vec = [0]*(math.ceil((end_time - start_time) / (mini_window - overlap)) + 1)

			if flag:
				vec = ret_vec_per_neuron(start_time, end_time, neuron, data)

			frame_vec.append(vec)
			
		from matplotlib import pyplot as plt
		plt.imshow(frame_vec, aspect = 'auto')
		plt.title("For frame at time: " + str(frame))
		plt.xlabel("Index of Window around the frame's occurence")
		plt.ylabel("Neuronal Channel")
		plt.draw()
		plt.show()

if __name__ == '__main__':
	basic_features()




