
%% process data using 0.1s bin
% try to reproduce the same results
% tau = .1; %100 ms time bins
% experiment_bounds = [0 42*60];
% ts_upsample = experiment_bounds(1):tau:experiment_bounds(end);
% ts_edges = [ts_upsample - tau/2 ts_upsample(end) + tau/2];
% resc = histc(processed_units(1).ts, ts_edges)
% comp_c = processed_units(1).spiketrain.spikeTrain;
% test = resc(1:end-1) - comp_c';
% sum(abs(test))
%%

%%% only output spiketrain and bin edges for further process and neuron
%%% region for further process
data_dir = "/media/yipeng/data/movie_2021/Movie_Analysis/data/"
%patient_ids = [431, 433, 435,436, 437, 439, 441, 442, 444, 445, 452]
tau = .05; %20 ms time bins
experiment_bounds = [0 42*60];
ts_upsample = experiment_bounds(1):tau:experiment_bounds(end);

for patient_id =[431, 433, 435,436, 439, 441, 444, 445, 452]
    datamat_dir = strcat(data_dir, num2str(patient_id))
    datamat_path = strcat(datamat_dir, "/final_clean_units_2021")
    load(datamat_path);
    ts_edges = [ts_upsample - tau/2 ts_upsample(end) + tau/2];
    number_neuron = length(units);
    number_timestamps = length(ts_edges);
    firing_mat = zeros(number_neuron, number_timestamps);
    patient(1).region = {};
    for i = 1:number_neuron
        firing_mat(i,:) = histc(units(i).ts, ts_edges);
        patient(1).region{end+1} = units(i).channelRegion;
    end
    patient(1).name = patient_id
    patient(1).firing = firing_mat
    patient(1).tbins = ts_edges   
    save(strcat(datamat_dir ,'/clean_data.mat'), 'patient')
end


%%% only output spiketrain and bin edges for further process and neuron
%%% region for further process
data_dir = "/media/yipeng/data/movie_2021/Movie_Analysis/data/"
%patient_ids = [431, 433, 435,436, 437, 439, 441, 442, 444, 445, 452]
tau = .05; %20 ms time bins
max_time_val = 4.6998e+03 + 3 % add additional 3 seconds 
experiment_bounds = [42*60 max_time_val];
ts_upsample = experiment_bounds(1):tau:experiment_bounds(end);
ts_edges = [ts_upsample - tau/2 ts_upsample(end) + tau/2];
for patient_id =[431, 433, 435,436, 439, 441, 444, 445, 452]
    datamat_dir = strcat(data_dir, num2str(patient_id))
    datamat_path = strcat(datamat_dir, "/final_clean_units_2021")
    load(datamat_path)
    number_neuron = length(units);
    number_timestamps = length(ts_edges);
    firing_mat = zeros(number_neuron, number_timestamps);
    patient(1).region = {};
    for i = 1:number_neuron
        firing_mat(i,:) = histc(units(i).ts, ts_edges);
        patient(1).region{end+1} = units(i).channelRegion;
    end
    patient(1).name = patient_id;
    patient(1).firing = firing_mat;
    patient(1).tbins = ts_edges;
    save(strcat(datamat_dir ,'/clean_data2.mat'), 'patient')
end


%% for each patient, find the max value
%max_val = 0
% data_dir = "C:\Users\Tonmoy Monsoor\Google Drive\MovieProject\data\"
% for patient_id =[431, 433, 435,436, 437, 439, 441, 442, 444, 445, 452]
%     datamat_dir = strcat(data_dir, num2str(patient_id))
%     datamat_path = strcat(datamat_dir, "\processed_units")
%     load(datamat_path)
%     number_neuron = length(processed_units)
%     for i = 1:number_neuron
%         if max_val < processed_units(i).ts(end)
%             max_val = processed_units(i).ts(end)
%         end
%     end
% end
%%