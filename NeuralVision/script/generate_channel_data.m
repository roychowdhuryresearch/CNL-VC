data_dir = "/media/yipeng/data/movie_2021/Movie_Analysis/data/"
for patient_id =[431, 433, 435,436, 439, 441, 444, 445, 452]
 
    fields = {'channel_info','region_info'} 
    c = cell(length(fields),1);
    channel_reg_info = cell2struct(c,fields);
    
    datamat_dir = strcat(data_dir, num2str(patient_id))
    datamat_path = strcat(datamat_dir, "/final_clean_units_2021")
    load(datamat_path);
    number_neuron = length(units);
    for i = 1:number_neuron
        channel_reg_info(i).channel_info = units(i).channelNum;
        channel_reg_info(i).region_info = units(i).channelRegion;
    end
    save(strcat(datamat_dir ,'/channel_data.mat'), "channel_reg_info")
end
