import shutil
import os
import sys
def clean_folder(saved_fn):
    if not os.path.exists(saved_fn):
        os.mkdir(saved_fn)
    else:
        shutil.rmtree(saved_fn)
        os.mkdir(saved_fn)

output = "/media/yipeng/data/movie/Movie_Analysis/train_data_clean/terro"
group_list1 =  [744, 936, 1048, 1058, 1973, 2008, 3048, 3101, 560, 1635, 3971, 1617, 4294, 5867, 6931, 908 ,896,901,7406,352,5252,4790,3782,3710, 970, 916,958,3191,2141,2013,1998,1556,1448,1408,992,950,930,546, 543, 480,544 ] 
#Jack
#group_list1 =  [744, 936, 1048, 1058, 1973, 2008, 3048, 3101, 560, 1635, 3971, 1617, 4294, 5867, 6931, 908 ,896,901,7406,352,5252,4790,3782,3710, 970, 916,958,3191,2141,2013,1998,1556,1448,1408,992,950,930,546, 543, 480, 1821,544 ] 
#terro
group_list2 =  [954, 2027, 2151,765,4175,3104,3070,1433, 1610, 2996,2148,1985,1075,1037,961,931,929]
#chole
group_list3 =  [193,204,212,216,373,395,414,428,435,841, 854,1273, 1281, 1283, 1289, 1319,1927, 1938, 2109, 1032, 2366, 2404, 3186, 3438, 3667, 858, 834 ,4071,4515,4013,3007,2597,1891,1814,1805,1307,1294,1270,1263,793,741, 443, 404, 392, 389, 206] 
#bill
group_list4 =  [3567,266,270,334, 995, 1004, 1362, 1413, 2076, 2092, 2424, 2458, 2496, 2690, 331,343,5854,558,869,873,889,1616,4612,2497,1922,1023,894,562, 297]

clean_folder(output)
in_dir = "/home/yipeng/Desktop/movie_tracking/tracking/group_result_network"
for a in group_list2:
    folder = os.path.join(in_dir, str(a))
    size = len(os.listdir(folder))
    shutil.copytree(folder,os.path.join(output,str(a)+"_"+str(size)))
