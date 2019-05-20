from __future__ import division
from math import radians, cos, sin, asin, sqrt, exp
from datetime import datetime
from pyspark import SparkContext
sc = SparkContext(appName="lab_kernel")

def haversine(lon1, lat1, lon2, lat2):
	"""Calculate the great circle distance between two points on the
	earth (specified in decimal degrees)"""
	# convert decimal degrees to radians
	lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
	# haversine formula
	dlon = lon2 -lon1
	dlat = lat2 -lat1
	a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
	c = 2 * asin(sqrt(a))
	km = 6367 * c
	return km

h_distance = 100 #km
h_date = 4 # Days
h_time = 2*60*60 # Seconds
a = 58.4274 # Up to you
b = 14.826 # Up to you
date = "2013-08-04" # Up to you
times = ["04:00:00","06:00:00","08:00:00","10:00:00","12:00:00",
	"14:00:00","16:00:00","18:00:00","20:00:00","22:00:00","00:00:00"]
temp = [0]*len(times)
temp_sum = [0]*len(times)
temp_mul = [0]*len(times)

stations_file = sc.textFile("/user/x_krisi/data/stations.csv")
temps_file = sc.textFile("/user/x_krisi/data/temperature-readings.csv")
#temps_file = temps_file.sample(False,0.1)

stationLines = stations_file.map(lambda line: line.split(";"))
stations = stationLines.map(lambda x: (int(x[0]), float(x[3]),
	float(x[4])))

tempLines = temps_file.map(lambda line: line.split(";"))
temps = tempLines.map(lambda x: (int(x[0]),x[1],x[2],float(x[3])))

def gaussianKernelDist(data,coords, h):
	u = data.map(lambda x: (x[0],haversine(x[2],x[1],coords[0],coords[1])/h))
	k = u.map(lambda x: (x[0],exp(-(x[1]**2))))
	#print k.collect()
	return k

def gaussianKernelDate(x,date,h):
	diff_date = (datetime(int(x[0:4]),int(x[5:7]),int(x[8:10]))
		- datetime(int(date[0:4]),int(date[5:7]),int(date[8:10]))).days / h
	k = exp(-(diff_date**2))
	#print(k.collect())
	return k

def gaussianKernelTime(x,time,h):
	diff_time = (datetime(2000,1,1,int(x[0:2]),int(x[3:5]),int(x[6:8]))
		- datetime(2000,1,1,int(time[0:2]),int(time[3:5]),int(time[6:8]))).seconds / h
	k = exp(-(diff_time**2))
	#print k.collect()
	return k


def predict():
	k_dist = gaussianKernelDist(stations,[b,a],h_distance)
	k_dist_broadcast = k_dist.collectAsMap()
	stations_dist = sc.broadcast(k_dist_broadcast)

	#Filter on date
	filtered_dates = temps.filter(lambda x:
		(datetime(int(x[1][0:4]),int(x[1][5:7]),int(x[1][8:10]))
		<= datetime(int(date[0:4]),int(date[5:7]),int(date[8:10]))))
	filtered_dates.cache()

	for time in times:
		#Filter on time
		filtered_times = filtered_dates.filter(lambda x:
			((datetime(int(x[1][0:4]),int(x[1][5:7]),int(x[1][8:10]))
			== datetime(int(date[0:4]),int(date[5:7]),int(date[8:10])))) and
			(datetime(2000,1,1,int(x[2][0:2]),int(x[2][3:5]),int(x[2][6:8]))
			<= datetime(2000,1,1,int(time[0:2]),int(time[3:5]),int(time[6:8]))))


		kernel = filtered_times.map(lambda x: (stations_dist.value[x[0]],
			gaussianKernelDate(x[1],date,h_date),
			gaussianKernelTime(x[2],time,h_time),x[3]))

		k_sum = kernel.map(lambda x: (x[0] * x[1] * x[2],x[3]))
		k_sum = k_sum.map(lambda x: (x[0]*x[1],x[0]))
		k_sum = k_sum.reduce(lambda x,y: (x[0]+y[0],x[1]+y[1]))
		temp_sum[times.index(time)] = (time,k_sum[0]/k_sum[1])

predict()
print(temp_sum)
