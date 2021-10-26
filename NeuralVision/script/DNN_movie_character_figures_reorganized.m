%% Define paths

warning('off', 'MATLAB:LargeImage');

figure_width_single_col = 8.9;
figure_width_double_col = 18.3;
figure_width_triple_col = 24.7;

set(0,'defaultlinelinewidth', 1);
set(0, 'DefaultAxesLineWidth', 1);
set(0, 'DefaultAxesTickDir', 'out');
set(0, 'DefaultAxesBox', 'off');
set(0,'DefaultAxesFontSize', 9); %
set(0,'DefaultTextFontSize', 9);
set(0, 'DefaultAxesFontWeight', 'normal');
set(0, 'DefaultAxestickLength', [0.025, 0.025])

addpath(genpath('/data/Projects/Movie/Movie_Analysis_Repo/Paper_Decoding/Codes'))
path_to_figures_data = '/media/yipeng/data/movie_2021/Movie_Analysis/final_result_outputs';
model_types = {'CNN_multi_2_KLD', 'knockout_test_CNN_KLD', 'LSTM_multi_2_KLD', 'knockout_test_LSTM_KLD'};
models = {'LSTM', 'CNN'};
figures_path = '/media/yipeng/data/movie_2021/Movie_Analysis/Paper_Decoding/Figures';
paper_path = pwd;

%setting up colors
zgray = [.6 .6 .6];
temp_cmp = hsv(3) / 1.5;
temp_cmp(2,:) = temp_cmp(2,:)/1.3;
temp_cmp(4,:) = [160,82,149]/255;
temp_cmp(5,:) = [160, 82, 45] / 255;
cmp = temp_cmp;

cmp_light(1,:) = [212 176 213]/255;
cmp_light(2,:) = [200 250 220]/255;
cmp_light(3,:) = [143 220 240]/255;

sr_final_movie_data = 7.5;

iM = 1; %for LSTM, for supplementary CNN, change to 2

patientNums = {'p_431', 'p_433', 'p_435', 'p_436', 'p_439', 'p_441', 'p_442', 'p_444', 'p_445', 'p_452'}; 
%% figure 2 : plot : overall accuracy of the model
if exist('fig', 'var'); try close(fig); end; end %#ok<TRYNC>
[panel_pos, fig, all_ax] = panelFigureSetup(5, 6, ['a         b         c   ed   f'], ...
    figure_width_double_col, 1.4, 1.4 );
example_patient_tpfp = 4;

axes('position', panel_pos(6,:).*[.1 .9 9.2 4])
nn_structure = imread(fullfile(figures_path, 'LSTM.png'));
imshow(nn_structure)
axis tight

fig2b = load(fullfile(path_to_figures_data, models{iM}, 'figure2b.mat'));
movie_time = [0: size(fig2b.stats_label,2) - 1]/sr_final_movie_data;
y_pos = [1.55, 1.35, 1.15 .95]+.05;

for char_num = 1 : 4
    axes('position', panel_pos(16,:).*[1.05, y_pos(char_num), 8.2, .5])
    %label
    char_1 = find(fig2b.stats_label(char_num,:) == 0);
    for iF = 1 : length(char_1)
        ar = area([movie_time(char_1(iF))-1/sr_final_movie_data, movie_time(char_1(iF))+1/sr_final_movie_data], ...
            [.5, .5], 'facecolor', cmp(2,:), 'facealpha', .7, 'edgecolor', 'none');
        hold on
    end

    %tp
    char_1 = find(fig2b.stats_tpfp(char_num,:) == 1);
    for iF = 1 : length(char_1)
        scatter(movie_time(char_1(iF)), .2, 4, 'markerfacecolor', cmp(3,:), 'markerfacealpha', .5, 'markeredgecolor', 'none');
        hold on
    end

    %fp
    char_1 = find(fig2b.stats_tpfp(char_num,:) == 2);
    for iF = 1 : length(char_1)
        scatter(movie_time(char_1(iF)), .3, 4, 'markerfacecolor', cmp(1,:), 'markerfacealpha', .3, 'markeredgecolor', 'none');
        hold on
    end
    box off
    xlim([0 2500])
    ylim([0 .5])
    set(gca, 'xtick', 0:500:2500)
    if char_num ~= 4; set(gca, 'xticklabel', []); end
    set(gca, 'ytick', [])
    ylabel(strcat('C. ',num2str(char_num)))
    if char_num == 4
        text(100,  .5, 'Label', 'color', cmp(2,:))
        text(100,  .35, 'TP', 'color', cmp(3,:))
        text(100,  .2, 'FP', 'color', cmp(1,:))
    end
    
    if char_num == 1
        text(650, .7, 'Character decoding for example participant', 'color', 'k', 'fontweight','bold')
    end
end
xlabel('Time (s)')

% patient 1 confusion matrix
fig2c = load(fullfile(path_to_figures_data, models{iM}, 'figure2c.mat'));
confusion_cmp = colormap(othercolor('Blues7'));
colormap(confusion_cmp)
for char_num = 1 : 4
    axes('position', panel_pos(20+char_num,:).*[.95 - char_num/50 .95 1.1 1.1])
    this_matrix = squeeze(fig2c.character_confusion_mat(char_num,1:2,1:2));
    this_matrix = 100 * this_matrix ./ sum(this_matrix, 2);
    h = heatmap(this_matrix(1:2, 1:2))
    h.Title = strcat('C. ', num2str(char_num))
    if char_num ==1
        h.YDisplayLabels = {'Y', 'N'};
    else
        h.YDisplayLabels = {'', ''};
    end
    %h.XDisplayLabels = {'Y', 'N'};
    h.XDisplayLabels ={'Y_{pred}', 'N_{pred}'};
%     box off
    if char_num ~=4
        h.ColorbarVisible = 'off'
    end
%     if char_num == 4; cb = colorbar('location', 'east'); cb.Position = cb.Position + .08; 
%         cb.Position(3) = cb.Position(3) * .2;
%         cb.Position(4) = cb.Position(4) * .75;
%         cb.Position(2) = cb.Position(2) - .09;
%         cb.Ticks = 0:25:100;
%         ylabel(cb, 'Norm. confusion matrix')
%     end
end

% overall character accuracy
pt_colormap = othercolor('BrBG10', 10);
load(fullfile(path_to_figures_data, models{iM}, 'figure2def.mat'));
fig2def = stats;
%restructure the data probably
for iP = 1 : length(patientNums)
    this_patient = patientNums{iP};
all_character_accuracy(iP,:) = eval(strcat('fig2def','.',this_patient)).character_acc;
all_character_f1_score(iP,:) = eval(strcat('fig2def','.',this_patient)).character_f1_score;
end
axes('position', panel_pos(30,:).*[0.98 .6 1.5 1.25])
for iP = 1 : length(patientNums)
    for char_num = 1 : 4
        scatter(-.25 + .5*rand(1,1)+char_num, all_character_f1_score(iP, char_num), 15, 'markerfacecolor', pt_colormap(iP,:), ...
            'markeredgecolor', 'none', 'markerfacealpha', .5)
        hold on
    end
end
xlim([0 6])
set(gca,'xtick', 1:4)
set(gca, 'xticklabel', {'C.1', 'C.2', 'C.3', 'C.4'})
xtickangle(90)
ylabel('F-1 score')
ylim([0.6 1])

axes('position', panel_pos(25,:).*[0.98 .88 1.5 1.25])
for iP = 1 : length(patientNums)
    for char_num = 1 : 4
        scatter(-.25 + .5*rand(1,1)+ char_num, +100*all_character_accuracy(iP, char_num), 15, 'markerfacecolor', pt_colormap(iP,:), ...
            'markeredgecolor', 'none', 'markerfacealpha', .5)
        hold on
    end
    text(5, 100 - (iP-1)*4.5, strcat('Pt ',  num2str(iP)), ...
        'color', pt_colormap(iP,:)/1.25, 'fontsize', 8, 'fontweight', 'bold')
end
xlim([0 6])
set(gca,'xtick', 1:4)
set(gca, 'xticklabel', [])
xtickangle(20)
ylabel('Accuracy')
ylim([80 100])
set(gca,'ytick', 0:10:100)

% all confusion matrices:
% freezeColors;
%
for char_num = 1 : 4
    axes('position', panel_pos(25+char_num,:).*[.95 - char_num/50 .05 1.1 2])
    all_patients_cm = nan(10,2,2);
    for iP = 1 : length(patientNums)
        this_patient = patientNums{iP};
        this_matrix = (squeeze(eval(strcat('fig2def','.',this_patient)).character_confusion_mat(char_num,1:3,1:3)));
        this_matrix = 100 * this_matrix ./ sum(this_matrix, 2);
        all_patients_cm(iP,:,:)= (this_matrix(1:2, 1:2));
    end
    b = ThreeDBarWithErrorBars(squeeze(nanmean(all_patients_cm, 1)), 2*squeeze(nanstd(all_patients_cm,[], 1)));
    for k = 1:length(b)
        zdata = b(k).ZData;
        b(k).CData = zdata;
        b(k).FaceColor = 'interp';
    end
    set(gca,  'xtick', 1:2)
    set(gca, 'ytick', 1:2)
    if char_num ==1
        set(gca, 'yticklabel', {'Y', 'N'});
    else
        set(gca, 'yticklabel', []);
    end
    set(gca, 'xticklabel',{'Y_{pred}', 'N_{pred}'});
    box off

end
capstr = [];
% savefig(fullfile(figures_path, 'Figure_2_July13'), fig, true, capstr, 600);
%% figure 3: data : important regions analysis - loading data from patients
fig3a = load(fullfile(path_to_figures_data, models{iM}, 'figure3a.mat'));
fig3b = load(fullfile(path_to_figures_data, models{iM}, 'figure3b.mat'));
fig3c = load(fullfile(path_to_figures_data, models{iM}, 'figure3c.mat'));
fig3f = load(fullfile(path_to_figures_data, models{iM}, 'figure3f.mat'));

all_knockout_results = [];
for iP = 1:length(patientNums) 
    iP
    
    data= [];
    patient_knockout = eval(strcat('fig3b.', patientNums{iP}));
    patient_microregion_knockout = eval(strcat('fig3f.', patientNums{iP}));
    
    clear pt_regions pt_regions_ptnum
    for iR = 2 : size(patient_knockout.tag_unique, 1)
        pt_regions{iR-1} = strtrim((patient_knockout.tag_unique(iR,:)));  
        pt_regions_ptnum{iR-1} = patientNums{iP};
    end
    
     clear pt_mw_regions
    for iR = 1 : size(patient_knockout.tag_unique_mw, 1)
        pt_mw_regions{iR} = strtrim((patient_knockout.tag_unique_mw(iR,:)));  
    end
    [sorted_mw, sorted_mw_I] = sort(pt_mw_regions);
    
    data.final_regions_ptnum = pt_regions_ptnum;
    data.final_unique_regions = pt_regions;
    data.final_microwire_regions = sorted_mw;

    data.normalized_region_loss = patient_knockout.loss(2:end,:)/max(max(patient_knockout.loss(2:end, :)));
    data.final_region_loss = patient_knockout.loss(2:end,:);
    temp = patient_knockout.loss_mw(2:end,:);
    data.final_microwire_loss = temp(sorted_mw_I,:);
    
    temp = patient_microregion_knockout.region_loss;
    data.final_sum_region = temp(:, sorted_mw_I);
    temp = patient_microregion_knockout.sum_mircowire_loss;
    data.final_sum_mw = temp(:, sorted_mw_I);
    
    all_knockout_results = [all_knockout_results; data];
    
end
all_knockout_results = fixMovieRegionNames(all_knockout_results);

all_regions_ptnum =  arrayfun(@(x) x.final_regions_ptnum', all_knockout_results, 'uniformoutput', false);
all_regions_ptnum = cat(1,all_regions_ptnum{:});

all_regions =  arrayfun(@(x) x.final_unique_regions', all_knockout_results, 'uniformoutput', false);
all_regions = cat(1,all_regions{:});
all_regions = cellfun(@(x) x(2:end), all_regions, 'UniformOutput', false);

all_microwires =  arrayfun(@(x) x.final_microwire_regions(2:end)', all_knockout_results, 'uniformoutput', false);
all_microwires = cat(1, all_microwires{:});
all_microwires = cellfun(@(x) x(2:end), all_microwires, 'UniformOutput', false);
    
all_regions(strcmp(all_regions, 'ACV')) = {'A. Cingulate'};
all_regions(strcmp(all_regions, 'AC')) = {'A. Cingulate'};
all_regions(strcmp(all_regions, 'MC')) = {'M-P. Cingulate'};
all_regions(strcmp(all_regions, 'PC')) = {'M-P. Cingulate'};
all_regions(strcmp(all_regions, 'EC')) = {'Entorhinal'};
all_regions(strcmp(all_regions, 'MH-EC')) = {'Entorhinal'};
all_regions(strcmp(all_regions, 'AH')) = {'Hippocampus'};
all_regions(strcmp(all_regions, 'MH')) = {'Hippocampus'};
all_regions(strcmp(all_regions, 'PHG')) = {'Parahippocampal'};
all_regions(strcmp(all_regions, 'SUB')) = {'Subiculum'};
all_regions(strcmp(all_regions, 'SS')) = {'Supra Sylvian'};
all_regions(strcmp(all_regions, 'SSA')) = {'Supra Sylvian'};
all_regions(strcmp(all_regions, 'SSP')) = {'Supra Sylvian'};
all_regions(strcmp(all_regions, 'TO')) = {'Occipital'};
all_regions(strcmp(all_regions, 'O')) = {'Occipital'};
all_regions(strcmp(all_regions, 'VMPC')) = {'vm-PFC'};
all_regions(strcmp(all_regions, 'VMPFC')) = {'vm-PFC'};
all_regions(strcmp(all_regions, 'TP')) = {'Parietal'};
all_regions(strcmp(all_regions, 'A')) = {'Amygdala'};
all_regions(strcmp(all_regions, 'ST')) = {'Superior Temporal'};
all_regions(strcmp(all_regions, 'STA')) = {'Superior Temporal'};
all_regions(strcmp(all_regions, 'STP')) = {'Superior Temporal'};
all_regions(strcmp(all_regions, 'STG')) = {'Superior Temporal'};
all_regions(strcmp(all_regions, 'STGA')) = {'Parietal'};
all_regions(strcmp(all_regions, 'STGP')) = {'Parietal'};

all_microwires(strcmp(all_microwires, 'ACV')) = {'A. Cingulate'};
all_microwires(strcmp(all_microwires, 'AC')) = {'A. Cingulate'};
all_microwires(strcmp(all_microwires, 'MC')) = {'M-P. Cingulate'};
all_microwires(strcmp(all_microwires, 'PC')) = {'M-P. Cingulate'};
all_microwires(strcmp(all_microwires, 'EC')) = {'Entorhinal'};
all_microwires(strcmp(all_microwires, 'MH-EC')) = {'Entorhinal'};
all_microwires(strcmp(all_microwires, 'AH')) = {'Hippocampus'};
all_microwires(strcmp(all_microwires, 'MH')) = {'Hippocampus'};
all_microwires(strcmp(all_microwires, 'PHG')) = {'Parahippocampal'};
all_microwires(strcmp(all_microwires, 'SUB')) = {'Subiculum'};
all_microwires(strcmp(all_microwires, 'SS')) = {'Supra Sylvian'};
all_microwires(strcmp(all_microwires, 'SSA')) = {'Supra Sylvian'};
all_microwires(strcmp(all_microwires, 'SSP')) = {'Supra Sylvian'};
all_microwires(strcmp(all_microwires, 'TO')) = {'Occipital'};
all_microwires(strcmp(all_microwires, 'O')) = {'Occipital'};
all_microwires(strcmp(all_microwires, 'VMPC')) = {'vm-PFC'};
all_microwires(strcmp(all_microwires, 'VMPFC')) = {'vm-PFC'};
all_microwires(strcmp(all_microwires, 'TP')) = {'Parietal'};
all_microwires(strcmp(all_microwires, 'A')) = {'Amygdala'};
all_microwires(strcmp(all_microwires, 'ST')) = {'Superior Temporal'};
all_microwires(strcmp(all_microwires, 'STA')) = {'Superior Temporal'};
all_microwires(strcmp(all_microwires, 'STP')) = {'Superior Temporal'};
all_microwires(strcmp(all_microwires, 'STG')) = {'Superior Temporal'};
all_microwires(strcmp(all_microwires, 'STGA')) = {'Parietal'};
all_microwires(strcmp(all_microwires, 'STGP')) = {'Parietal'};

[all_regions, jR] = sort(all_regions);
[all_microwires, jM] = sort(all_microwires);

all_region_loss_norm = arrayfun(@(x) x.normalized_region_loss, all_knockout_results, 'uniformoutput', false);
all_region_loss_norm = cat(1, all_region_loss_norm{:});
all_region_loss_norm = all_region_loss_norm(jR,:);

all_microwire_loss = arrayfun(@(x) x.final_microwire_loss, all_knockout_results, 'uniformoutput', false);
all_microwire_loss = cat(1, all_microwire_loss{:});
all_microwire_loss = all_microwire_loss(jM,:)

all_region_loss= arrayfun(@(x) x.final_region_loss, all_knockout_results, 'uniformoutput', false);
all_region_loss = cat(1, all_region_loss{:});
all_region_loss = all_region_loss(jR,:);

all_sum_microwire_loss = arrayfun(@(x) x.final_sum_mw', all_knockout_results, 'uniformoutput', false);
all_sum_microwire_loss = cat(1, all_sum_microwire_loss{:});
all_sum_microwire_loss = all_sum_microwire_loss(jM,:);

all_sum_region_loss = arrayfun(@(x) x.final_sum_region', all_knockout_results, 'uniformoutput', false);
all_sum_region_loss = cat(1, all_sum_region_loss{:});
all_sum_region_loss = all_sum_region_loss(jM,:);

load('/data/Projects/Movie/Movie_Analysis_Repo/final_result_outputs/patient_performance.mat');
%% figure 3 : plot :  important regions analysis - loading data from patients
if exist('fig', 'var'); try close(fig); end; end %#ok<TRYNC>
[panel_pos, fig, all_ax] = panelFigureSetup2(2, 3, ['abc  e'], ...
    figure_width_triple_col, figure_width_double_col, 1.4, 1.4 );
example_patient_ko = 9;

% new_cmp = (othercolor('BuDRd_12'));
% new_cmp = new_cmp(end:-1:1, :);
% half_cmp = new_cmp(end/2:end, :);
% colormap(half_cmp)

new_cmp = colormap(othercolor('GnBu7', 30));

this_ko = all_knockout_results(example_patient_ko);
this_region_loss = this_ko.final_region_loss';
this_microwire_loss = this_ko.final_microwire_loss';
this_sum_mw_loss = this_ko.final_sum_mw;
this_sum_region_loss = this_ko.final_sum_region;
unique_regions = this_ko.final_unique_regions;
unique_regions_mw =  this_ko.final_microwire_regions;

axes('position', panel_pos(1,:).*[1 1 .9 1])
imagesc(this_region_loss); shading flat
caxis([0 .12])
set(gca,'ytick', 1:4)
set(gca, 'yticklabel', {'C.1', 'C.2', 'C.3', 'C.4'})
ylabel('Region knockout')
freezeColors;
set(gca,'xtick', 1:length(unique_regions))
set(gca, 'xticklabel', unique_regions)
xtickangle(30)

cb = colorbar('location', 'east'); 
cb.Position = cb.Position + .09; 
cb.Position(3) = cb.Position(3) * .2;
cb.Position(4) = cb.Position(4) * .75;
cb.Position(2) = cb.Position(2) - .1;
cb.Position(1) = cb.Position(1);
cb.Ticks = 0:.06:.12;
ylabel(cb, 'Normalized KLD loss')

box off

axes('position', panel_pos(3,:).*[.95 1.05 .9 1])
imagesc(this_microwire_loss); shading flat
caxis([0 .3])
set(gca,'ytick', 1:4)
set(gca, 'yticklabel', {'C.1', 'C.2', 'C.3', 'C.4'})
[temp_regions, temp_I] = unique(unique_regions_mw);
set(gca,'xtick', temp_I)
set(gca, 'xticklabel', temp_regions); %, 'fontsize', 7)
xtickangle(45)
ylabel('Electrode knockout')
freezeColors;

cb = colorbar('location', 'east'); 
cb.Position = cb.Position + .09; 
cb.Position(3) = cb.Position(3) * .2;
cb.Position(4) = cb.Position(4) * .75;
cb.Position(2) = cb.Position(2) - .1;
cb.Position(1) = cb.Position(1);
cb.Ticks = 0:.15:.3;
ylabel(cb, 'Normalized KLD loss')

box off

y_pos = [3.5, 2.525, 1.55 .65]

[unique_sorted_regions_mw, unique_sorted_mw_I] = unique(unique_regions_mw);
this_sum_region_loss = this_sum_region_loss(:, unique_sorted_mw_I);
this_sum_mw_loss = this_sum_mw_loss(:, unique_sorted_mw_I);

for char_num = 1 : 4
    axes('position', panel_pos(5,:).*[1, y_pos(char_num), .9, .2])
    y = this_sum_region_loss(char_num,:) - this_sum_mw_loss(char_num,:);
    x = 1 : length(unique_sorted_regions_mw);
    h = plot(x, y);
    cd = colormap(new_cmp);
    cd = interp1(linspace(min(y),max(y),length(cd)),cd,y); % map color to y values
    cd = uint8(cd'*255); % need a 4xN uint8 array
    cd(4,:) = 255; % last column is transparency
    hold  on
    set(h.Edge,'ColorBinding','interpolated','ColorData',cd)
    freezeColors;
    box off
    set(gca,'xtick', 1:length(unique_sorted_mw_I))
    set(gca,'xticklabel', [])
    if char_num == 4
        set(gca, 'xticklabel', unique_sorted_regions_mw)
        xtickangle(30)
    end
%     ylim([-.2 1])
%     set(gca, 'ytick', 0:1)
    
    ylabel(strcat('C. ', num2str(char_num)))
    if char_num == 1; text(2, 1.4, 'KLD loss_{region }- \Sigma KLD Loss_{electrode}');end
    
    if char_num == 1; ylim([0 1.2]); set(gca,'ytick', 0:1.2:1.2);
        text(-.05, 1.6, 'd', 'fontweight','bold', 'fontsize', 9)
    elseif char_num == 2 | char_num == 4; ylim([0 .6]); set(gca,'ytick', 0:.5:.5);
    elseif char_num == 3; ylim([0 .8]); set(gca,'ytick', 0:.8:.8);end
end


% group analysis
axes('position', panel_pos(2,:) .*[1.15 .95 .9 1.2])    
regions_forDisplay = unique(all_regions);
regions_forDisplay = regions_forDisplay([1:8, 10:12]);

hold on
clear these_losses
for iR = 1 : length(regions_forDisplay)
        this_loss = all_region_loss_norm(strcmp(all_regions, regions_forDisplay{iR}),:);
        B = bar(iR, nanmedian(this_loss(:)),'facecolor', zgray , 'edgecolor', 'none');
        B.FaceAlpha = .5;
        these_losses{iR} = this_loss(:);
end

for iR = 1 : length(regions_forDisplay)
    for char_num = 1 : 4
        this_loss = all_region_loss_norm(strcmp(all_regions, regions_forDisplay{iR}),char_num);
        scatter(-2/8 + iR*ones(size(this_loss(:)))+char_num/8, this_loss(:),10,'markerfacecolor', cmp(char_num,:), 'markeredgecolor', 'none', ...
            'markerfacealpha', .5)
    end
end
box off    
set(gca,'xtick', 1:length(regions_forDisplay))
% set(gca, 'xticklabel', [])
set(gca, 'xticklabel', regions_forDisplay)
xtickangle(25)
xlim([0 12])
ylim([0 1.3])
set(gca, 'ytick', 0:.5:1)
h = ylabel('Relative loss (region knockout)');
set(h,'position', get(h,'position')+[-.5, -.8  0])
text(1, 1.3, '* C.1', 'color', cmp(1,:))
text(4.5, 1.3, '* C.2', 'color', cmp(2,:))
text(8, 1.3, '* C.3', 'color', cmp(3,:))
text(11.5, 1.3, '* C.4', 'color', cmp(4,:))

% compare the losses in regions
clear p_regions
for i = 1 : length(regions_forDisplay)
    for j = 1 : length(regions_forDisplay)
%         if j>i
            p_regions(i,j) = ranksum(these_losses{i}, these_losses{j});
%         end
    end
end
%
axes('position', panel_pos(4,:) .*[1.15 .93 .9 .8])    
low_performers = patientNums(overall_performance <= nanmedian(overall_performance)); %(overall_performance>0)));
high_performers = patientNums(overall_performance > nanmedian(overall_performance)); %(overall_performance>0)));
for iR = 1 : length(regions_forDisplay)
    valid = strcmp(all_regions, regions_forDisplay{iR}) | ismember(all_regions_ptnum, high_performers);
    this_loss_high = all_region_loss_norm(valid,:);
    B = bar(iR-.15, nanmedian(this_loss_high(:)),'facecolor', cmp(1,:) , 'edgecolor', 'none','barwidth', .3);
    B.FaceAlpha = .5;
    hold on

    valid = strcmp(all_regions, regions_forDisplay{iR}) | ismember(all_regions_ptnum, low_performers);
    this_loss_low = all_region_loss_norm(valid,:);
    B = bar(iR+.15, nanmedian(this_loss_low(:)),'facecolor', cmp(3,:) , 'edgecolor', 'none', 'barwidth', .3);
    B.FaceAlpha = .5;
%     [~, p]=ttest2(this_loss_low(:), this_loss_high(:));
    [p_raw]=ranksum(this_loss_low(:), this_loss_high(:));
    p=p_raw*12;
    if p<.05 & p>.01
    text(iR,.25,'*','fontsize', 10)
    elseif p<.01 & p>.001
    text(iR,.25,'**','fontsize', 10)
    elseif p<.001
    text(iR,.25,'***','fontsize', 10)
    end
    p_low_high(iR) = p_raw;
    
%     length(this_loss_low(:))
%     length(this_loss_high(:))
%     pause
end
[corrected_p_low_high, h]=bonf_holm(p_low_high,0.05);

set(gca,'xtick', 1:length(regions_forDisplay))
set(gca, 'xticklabel',  [])
xlim([0 12])
set(gca,'ytick', 0:.1:.3)
box off
text(.5,.34,'High performer', 'color', cmp(1,:),  'fontweight','bold', 'fontsize', 7)
text(.5,.31,'Low performer', 'color', cmp(3,:), 'fontweight','bold', 'fontsize', 7)

axes('position', panel_pos(6,:) .*[1.15 1.35 .9 1])    
% bin_edges = [-1:.1:-.5, -.4:.01:.4, .5:.1:2];
bin_edges = [-1:.025:2];
loss_diff = nan(length(regions_forDisplay), length(bin_edges));
hold on
clear percent_whole
for iR = 1 : length(regions_forDisplay)
    valid_region = strcmp(all_regions, regions_forDisplay{iR});
    valid_microwire = strcmp(all_microwires, regions_forDisplay{iR});
        this_loss = all_sum_region_loss(valid_microwire,:) - all_sum_microwire_loss(valid_microwire,:);
        [~,ind]=unique(this_loss(:,1));
        this_loss = this_loss(ind,:);
        [phat, pci] = binofit(sum(this_loss(:)>0),length(this_loss(:)));
        percent_whole(iR,1:3) = [phat, pci];
        [n,x] = hist(this_loss(:), bin_edges);
        loss_diff(iR, :) = n;
end

% colormap(hot)
imagesc(1:length(regions_forDisplay),bin_edges, loss_diff'); shading interp
box off    
set(gca,'xtick', 1:length(regions_forDisplay))
xlim([0 12])
ylabel({'KLD loss', '(region - electrode)'})
set(gca, 'xticklabel', regions_forDisplay)
xtickangle(25)
ylim([-1 1.2])
set(gca,'ytick', -1:.5:2, 'yticklabel',{'-1', '', '0', '', '1', '' ,''})
caxis([0 6])

cb = colorbar('location', 'westoutside'); 
cb.Position = cb.Position + .09; 
cb.Position(3) = cb.Position(3) * .2;
cb.Position(4) = cb.Position(4) * .5;
cb.Position(2) = cb.Position(2) - .09;
cb.Position(1) = cb.Position(1) - .21;
cb.Ticks = 0:3:6
ylabel(cb, 'Count')
cb.AxisLocation  = 'out';

capstr = [];
% savefig(fullfile(figures_path, 'Figure_3_July14'), fig, true, capstr, 600);

%% this is to test if the percentages are different among regions

figure()
for iR = 1 : length(regions_forDisplay)
        bar(iR, percent_whole(iR,1) , 'facecolor', cmp(2,:))
        hold on
        errorbar(iR, percent_whole(iR,1), percent_whole(iR,1) - percent_whole(iR,2),...
        percent_whole(iR,3) - percent_whole(iR,1), 'k')
end
set(gca,'xtick', 1:length(regions_forDisplay))
set(gca, 'xticklabel', regions_forDisplay)
xtickangle(25)

%% Retrain important - nonimportant stats:
retrain_results = load(fullfile(path_to_figures_data, models{iM}, 'important_stats_LSTM_1.mat'))
p_retrain = ranksum(retrain_results.important, retrain_results.unimportant)
%% figure 4 : data : memory part
fig4a = load(fullfile(path_to_figures_data, models{iM}, 'figure4a.mat'));
fig4c = load(fullfile(path_to_figures_data, models{iM}, 'figure4c.mat'));
folds = {'fold_0', 'fold_1', 'fold_2', 'fold_3', 'fold_4'};
memory_patients = {'431', '435', '436', '441', '442'};
all_memory_results = [];
for iP = 1:length(memory_patients)  
    iP
    data = [];
    this_patient_memory= eval(strcat('fig4c.p_', memory_patients{iP}));

    for iF = 1 : length(folds)
        data.start(iF,:,:,:) = eval(strcat('this_patient_memory.kfold_prediction_4sec.', folds{iF}, '.start'));
        data.end(iF,:,:,:) = eval(strcat('this_patient_memory.kfold_prediction_4sec.', folds{iF}, '.end'));
        data.response_end(iF,:,:,:) = eval(strcat('this_patient_memory.kfold_prediction_4sec.', folds{iF}, '.response_end'));
    end
%     trigs = load(fullfile('/data/Projects/Movie/Raw_Data/',memory_patients{iP},...
%         strcat(memory_patients{iP}, 'memtest1/psychophysics/EXP_',memory_patients{iP}, 'memtest1.mat')));
    data.trigs = trigs;
    data.clip_character = this_patient_memory.character_each_clip;
    data.clip_episode = this_patient_memory.episode_each_clip;
    data.responses = this_patient_memory.responses;
    all_memory_results = [all_memory_results; data];
        
end

first_step_associations = [1.0, 0.672, 0.0, 0.776, 0.557, 1.0, 0.187,0.257,...
0.0,0.133,1.0,0.035,0.507,0.202,0.039, 1.0];
second_step_associations =  [1.0, 0.8282041362361431, 0.15553410733277281, 0.9482255104964116; ...
    0.6870705166249167, 1.0, 0.1969343082497601, 0.6953589380435576; ...
    0.09176604242284847, 0.14005990447377425, 1.0,0.06897894228908416;...
    0.6189591399214647, 0.5471362243336229, 0.07631510647995454,  1.0];

all_patients_prediction = [];
all_patients_prediction_exp = [];
all_patients_prediction_end = [];
all_patients_prediction_exp_end = [];
all_patients_prediction_response_end = [];
all_patients_prediction_exp_response_end = [];
all_patients_clip_in = [];
all_patients_charac_in = [];
all_patients_seen = [];
all_patients_pt_number = [];
all_patients_char_number= [];


for ip = 1:5
    data = all_memory_results(ip);
    response = data.responses; %[data.trigs.trial_struct.resp_answer];
    episode_in = data.clip_episode(:) == 1;
    seen = (response == 1 |response == 2)';
    correct = (response == 1 |response == 2)' & episode_in;
    this_memory_start = squeeze(nanmean(data.start,1));
    this_memory_end = squeeze(nanmean(data.end,1));
    this_memory_response_end = squeeze(nanmean(data.response_end,1));
time_axis =  linspace(-2,2,30);
after = 16:30;
before = 1:14;
for iC = 1 : 4
        char_in = data.clip_character(:,iC)==1;
        temp_start = squeeze(nansum(this_memory_start(:,iC,after),3));
        temp_end = squeeze(nansum(this_memory_end(:,iC,after),3));
        temp_response_end = squeeze(nansum(this_memory_response_end(:,iC,before),3));
        temp_start_exp = squeeze(this_memory_start(:,iC,:));
        temp_end_exp = squeeze(this_memory_end(:,iC,:));
        temp_response_end_exp = squeeze(this_memory_end(:,iC,:));
        pred_all = [temp_start];
        pred_all_exp = [temp_start_exp];
        pred_all_end = [temp_end];
        pred_all_exp_end = [temp_end_exp];
        pred_all_response_end = [temp_response_end];
        pred_all_exp_response_end = [temp_response_end_exp];
        clip_in = [episode_in(:)];
        all_char_in = [char_in];
        all_seen = [seen(:)];
        char_number = iC * [ones(size(temp_start))];
        pt_number = ip * [ones(size(temp_start))];

        all_patients_prediction = [all_patients_prediction; pred_all];
        all_patients_prediction_exp = [all_patients_prediction_exp; pred_all_exp];
        all_patients_prediction_end = [all_patients_prediction_end; pred_all_end];
        all_patients_prediction_exp_end = [all_patients_prediction_exp_end; pred_all_exp_end];
        all_patients_prediction_response_end = [all_patients_prediction_response_end; pred_all_response_end];
        all_patients_prediction_exp_response_end = [all_patients_prediction_exp_response_end; pred_all_exp_response_end];
        all_patients_clip_in = [all_patients_clip_in;clip_in];
        all_patients_charac_in = [all_patients_charac_in; all_char_in];
        all_patients_seen = [all_patients_seen; all_seen];
        all_patients_pt_number = [all_patients_pt_number; pt_number];
        all_patients_char_number= [all_patients_char_number; char_number];
end
    
end

final_pred = all_patients_prediction >= 0.5;
dsa = table(final_pred , all_patients_seen, all_patients_charac_in, all_patients_clip_in);
mdlspec = 'final_pred ~ all_patients_charac_in + all_patients_seen + all_patients_clip_in';
% mdlspec = 'final_pred ~ all_patients_seen*all_patients_charac_in'; % - all_patients_seen:all_patients_charac_in';% - all_seen:all_char_in:clip_mode';
mdl = fitglm(dsa, mdlspec, 'Distribution', 'binomial')

final_pred_end = all_patients_prediction_end >= 0.5;
dsa_end = table(final_pred_end , all_patients_seen, all_patients_charac_in, all_patients_clip_in);
mdlspec_end = 'final_pred_end ~ all_patients_charac_in + all_patients_seen + all_patients_clip_in';
mdl_end = fitglm(dsa_end, mdlspec_end, 'Distribution', 'binomial') 

final_pred_response_end = all_patients_prediction_response_end >= 0.5;
dsa_response_end = table(final_pred_response_end , all_patients_seen, all_patients_charac_in, all_patients_clip_in);
mdlspec_response_end = 'final_pred_response_end ~ all_patients_charac_in + all_patients_seen + all_patients_clip_in';
mdl_response_end = fitglm(dsa_response_end, mdlspec_response_end, 'Distribution', 'binomial') 

% clf
% cond1 = all_patients_charac_in>0;
% shadedErrorBar(1:30, nanmean(all_patients_prediction_exp_end(cond1,:),1), ...
%     nanstd(all_patients_prediction_exp_end(cond1,:), 1)/sqrt(sum(cond1)), 'r', 1);
% hold on
% cond2 = ~cond1;
% shadedErrorBar(1:30, nanmean(all_patients_prediction_exp_end(cond2,:),1), ...
%     nanstd(all_patients_prediction_exp_end(cond2,:), 1)/sqrt(sum(cond2)), 'b', 1);
%
fig4b_percentage = load(fullfile(path_to_figures_data, models{iM}, 'figure4b_percentage.mat'));
fig4b_size = load(fullfile(path_to_figures_data, models{iM}, 'figure4b_size.mat'));
patient_memory_accuracy_percentage = [];
patient_memory_accuracy_size= [];
for iP = 1:length(memory_patients)  
    iP
    data = [];
    this_patient_percentage = eval(strcat('fig4b_percentage.p_', memory_patients{iP}));
    this_patient_size = eval(strcat('fig4b_size.p_', memory_patients{iP}));
    patient_memory_accuracy_percentage = [patient_memory_accuracy_percentage this_patient_percentage(:,2)/this_patient_percentage(1,2)];
    patient_memory_accuracy_size = [patient_memory_accuracy_size this_patient_size(:,2)/this_patient_size(1,2)];
end
patient_memory_percentange = this_patient_percentage(:,1);

fig4d= load(fullfile(path_to_figures_data, models{iM}, 'figure4d.mat'));
fig4e = load(fullfile(path_to_figures_data, models{iM}, 'figure4e.mat'));
memory_viewing = [];
memory_response= [];
memory_viewing_trimmed= [];
memory_response_trimmed = [];

for iP = 1:length(memory_patients)  
    first_step_all = fig4d.reference;
    this_patient_one_step = eval(strcat('fig4d.p_', memory_patients{iP}));
    temp = (this_patient_one_step{1}.patient_acc_stats(1,:));
    temp=reshape(temp,4,4)';
    memory_viewing = [memory_viewing; temp(:)'];
    temp([1,6,11,16])=[];
    memory_viewing_trimmed=[memory_viewing_trimmed,temp];
    
    this_patient_two_step= eval(strcat('fig4e.p_', memory_patients{iP}));
    temp = (this_patient_two_step{1}.patient_acc_stats(1,:));
    temp=reshape(temp,4,4)';
    memory_response = [memory_response; temp(:)'];
    temp([1,6,11,16])=[];
    memory_response_trimmed=[memory_response_trimmed,temp];
    first_step_all = fig4d.reference;
    second_step_all = fig4e.reference;
end
temp_first_step = first_step_all;
temp_first_step([1,6,11,16]) = [];
first_step_all_trimmed = repmat(temp_first_step,5,1);
temp_second_step = second_step_all;
temp_second_step([1,6,11,16]) = [];
second_step_all_trimmed = repmat(temp_second_step,5,1);

%comparing conditional probs clips and response
[h_cp, p_cp] = signrank(memory_response_trimmed(:)- memory_viewing_trimmed(:))
%% 
figure(1)
clf
for i = 1 : 5
    subplot(1,5,i)
    plot(first_step_all, memory_viewing(i,:), '*')
    temp = zeros(size(memory_viewing(i,:)));
    temp = memory_viewing(i,:);
    temp = temp(:);
    [r,p]= corr(first_step_all(temp~=1)', memory_viewing(i,temp~=1)', 'type','pearson');
    title(strcat(num2str(r), '-', num2str(p)))
    axis square
end
sgtitle('clip viewing - first step')
figure(2)
clf
for i = 1 : 5
    subplot(2,5,i)
    plot(first_step_all, memory_response(i,:),'*')
    temp=zeros(size(memory_response(i,:)));
    temp = memory_response(i,:);
    temp = temp(:);
    [r,p]= corr(first_step_all(temp~=1)', memory_response(i,temp~=1)', 'type','pearson')
    title(strcat(num2str(r),'-', num2str(p)))
    axis square
    subplot(2,5,i+5)
    plot(second_step_all, memory_response(i,:),'*')
    [r,p]= corr(second_step_all(temp~=1)', memory_response(i,temp~=1)', 'type','pearson')
    title(strcat(num2str(r), '-', num2str(p)))
    axis square
end
sgtitle('response time - both')
%% figure 4: plot : memory part
if exist('fig', 'var'); try close(fig); end; end %#ok<TRYNC>
[panel_pos, fig, all_ax] = panelFigureSetup2(2, 3, ['abcdef'], ...
    figure_width_double_col, figure_width_double_col, 1.4, 1.4 );

cmp_activation = colormap(othercolor('GnBu7', 30));

%---- panel A: example character
ic = 1;
ip =5;
clip_in = 1;
cond1 = all_patients_charac_in == 1 & all_patients_char_number==ic & all_patients_clip_in== clip_in & all_patients_pt_number == ip;
cond2 = all_patients_charac_in == 0 & all_patients_char_number==ic & all_patients_clip_in== clip_in & all_patients_pt_number == ip;

axes('position', panel_pos(1,:).*[1 1 .4 1])
i = 1;
this_count = sum(eval(strcat('cond',num2str(i))));
pcolor(time_axis, 1:this_count, ...
        all_patients_prediction_exp(eval(strcat('cond',num2str(i))),:))
shading flat
ylim([1 this_count-1])
xlim([0 2])
caxis([0 .5])
ylabel('Clip number')
set(gca, 'ytick', 0:this_count:this_count)
set(gca, 'xtick', 0:1:2)
set(gca, 'xticklabel', {'0', '1', '2'})
h = xlabel('Time from clip onset (s)');
set(h, 'position', get(h,'position')+[1 0 0])
title('Character in')


axes('position', panel_pos(1,:).*[3.5 1 .4 1])
i = 2;
rng(36)
cond2_ind = find(cond2);
random_perm = randperm(length(cond2_ind), this_count)';
ind_cond2 =  cond2_ind(random_perm);
% this_count = length(eval(strcat('ind_cond',num2str(i))));
pcolor(time_axis, 1:this_count, ...
        all_patients_prediction_exp(eval(strcat('ind_cond',num2str(i))),:))
shading flat
ylim([1 this_count-1])
xlim([0 2])
caxis([0 .5])
set(gca, 'ytick', 0:this_count:this_count, 'yticklabel', [])
set(gca, 'xtick', 0:1:2)
% set(gca, 'xticklabel', {'0', '1'})
% xlabel('Time (s)')
title('Character out')
cb = colorbar('location', 'east'); cb.Position = cb.Position + .08; 
cb.Position(3) = cb.Position(3) * .2;
cb.Position(4) = cb.Position(4) * .75;
cb.Position(2) = cb.Position(2) - .09;
cb.Ticks = 0:.5:.5;
h = ylabel(cb, 'Avg. model activation');
set(h, 'position', get(h,'position')+[1 0 0])

freezeColors();
% --- panel B: memory decoding accuracy as a function of size and duration
% of the character
axes('position', panel_pos(2,:).*[1.1 1.05 .45 1])
shadedErrorBar(patient_memory_percentange, nanmean(patient_memory_accuracy_size,2), ...
    nanstd(patient_memory_accuracy_size,[],2)/sqrt(length(memory_patients)))
box off
xlabel({'Character size','frame %'})
ylabel('Relative accuracy')
ylim([.98 1.1])
set(gca,'ytick',1:.05:1.1, 'yticklabel', {'1', '','1.1'})

axes('position', panel_pos(2,:).*[1.48 1.05 .45 1])
shadedErrorBar(patient_memory_percentange, nanmean(patient_memory_accuracy_percentage,2),...
    nanstd(patient_memory_accuracy_percentage,[],2)/sqrt(length(memory_patients)))
box off
xlabel({'Character presence','clip %'})
ylim([.98 1.1])
set(gca,'ytick',1:.05:1.1, 'yticklabel',[])

% --- panel C example patient 4 conditions
cmp_activation = colormap(othercolor('GnBu7', 40));
ip =1;
axes('position', panel_pos(3,:).*[1 1 1 1])
cond1 = all_patients_charac_in == 1 & all_patients_seen == 1 & all_patients_pt_number == ip;
cond2 = all_patients_charac_in == 1 & all_patients_seen == 0 & all_patients_pt_number == ip;
cond3 = all_patients_charac_in == 0 & all_patients_seen == 1 & all_patients_pt_number == ip;
cond4 = all_patients_charac_in == 0 & all_patients_seen == 0 & all_patients_pt_number == ip;

cmp_activation = cmp_activation(end:-1:1,:);
for i = 4 : -1: 1
shadedErrorBar(time_axis, nanmean(all_patients_prediction_exp(eval(strcat('cond',num2str(i))),:)),...
    nanstd(all_patients_prediction_exp(eval(strcat('cond',num2str(i))),:))/...
    (sum(eval(strcat('cond',num2str(i))))), {'color', cmp_activation(10*(i-1)+1,:)/1.2}, .8)
hold on
% pause
end
text_y = 0.13
text_y_step = .008;
text( -.2, text_y-text_y_step, 'Character in & Seen', 'color', cmp_activation(10*(1-1)+1,:)/1.5)
text( -.2, text_y-2*text_y_step, 'Character in & not Seen', 'color', cmp_activation(10*(2-1)+1,:)/1.5)
text( -.2, text_y-3*text_y_step, 'Character out & Seen', 'color', cmp_activation(10*(3-1)+1,:)/1.5)
text( -.2, text_y-4*text_y_step, 'Character out &  not Seen', 'color', cmp_activation(10*(4-1)+1,:)/1.5)
xlim([-.3 2])
box off
ylim([0 .12])
set(gca,'ytick', 0:.06:.12, 'yticklabel', 0:.5:1)
ylabel('Model activation (au)')
xlabel('Time from clip onset(s)')

%--- panel c : GLM results
cmp_activation = colormap(othercolor('GnBu7', 40));

axes('position', panel_pos(4,:).*[1.08 1.05 1 1])
cond1 = all_patients_charac_in == 1 & all_patients_seen == 1;% & all_patients_pt_number == ip;
cond2 = all_patients_charac_in == 1 & all_patients_seen == 0;% & all_patients_pt_number == ip;
cond3 = all_patients_charac_in == 0 & all_patients_seen == 1;% & all_patients_pt_number == ip;
cond4 = all_patients_charac_in == 0 & all_patients_seen == 0;% & all_patients_pt_number == ip;

cmp_activation = cmp_activation(end:-1:1,:);
for i = 4 : -1: 1
br = bar(i, mean(final_pred(eval(strcat('cond',num2str(i))))));
br.FaceColor = cmp_activation(10*(i-1) + 1,:)/1.1;
hold on
errorbar(i, mean(final_pred(eval(strcat('cond',num2str(i))))), ... 
    std(final_pred(eval(strcat('cond',num2str(i))))) / sqrt(length(final_pred(eval(strcat('cond',num2str(i)))))),'k');
% pause
end
box off
set(gca, 'xtick', 1:4, 'xticklabel', {'Char. in & seen', 'Char. in & ~seen', 'Char. out & seen', 'Char. out & ~seen'})
xtickangle(25)
ylabel('Model activation')
ylim([0 .7])
set(gca, 'ytick', 0:.35:.7)

%
% -- panel e: example conditional probabilities
axes('position', panel_pos(5,:).*[.4 .8 .8 .8])
imagesc(reshape(first_step_all(1:16), [4,4]))
axis square
box off
ylabel({'Characters', 'Cond. Prob.'})
title('During movie')
set(gca,'xtick', 1:4)
set(gca, 'xticklabel', {'C. 1', 'C. 2', 'C. 3', 'C. 4'})
set(gca,'ytick', 1:4)
set(gca, 'yticklabel', {'C. 1', 'C. 2', 'C. 3', 'C. 4'})
xtickangle(90)
cb = colorbar('location', 'northoutside'); 
cb.Position = cb.Position + .08; 
cb.Position(4) = cb.Position(4) * .2;
% cb.Position(4) = cb.Position(3) * .75;
cb.Position(2) = cb.Position(2) + .01;
cb.Ticks = 0:.5:1;
cb.TickLabels = {'0','','1'};
cb.TickDirection = 'out';
h = ylabel(cb, 'Cond. prob.');
set(h, 'position', get(h,'position')+[0 -1 0]);

x_pos = [3.5 6.9 10.9];

i = 2;
axes('position', panel_pos(5,:).*[x_pos(1) .8 .8 .8])
imagesc(reshape(memory_viewing(i,:), [4,4]))
axis square
box off
set(gca, 'xtick', 1:4, 'xticklabel', [])
ylabel({'Model Cond. Prob.', 'clip viewing'}); 
set(gca,'yticklabel', [])
title(strcat('Pt. ', num2str(i)))
set(gca,'xtick', 1:4)
set(gca, 'xticklabel', {'C. 1', 'C. 2', 'C. 3', 'C. 4'})
xtickangle(90)

% i = 2;
% axes('position', panel_pos(5,:).*[x_pos(1) .8 .5 .5])
% imagesc(reshape(response_cond_all(patient_cond_all ==i), [4,4]))
% axis square
% box off
% set(gca, 'xtick', 1:4)
%  ylabel({'Model Cond. Prob.', 'Response'}); 
% set(gca,'yticklabel', [])
% title(strcat('Pt. ', num2str(i)))
% set(gca,'xtick', 1:4)
% set(gca, 'xticklabel', {'Ch. 1', 'Ch. 2', 'Ch. 3', 'Ch. 4'})
% xtickangle(90)



axes('position', panel_pos(5,:).*[x_pos(2) 1 .75 .75])
scatter(first_step_all, memory_viewing(i,:), 50*ones(size(first_step_all)), ...
    'markerfacecolor', cmp(i,:), 'markeredgecolor', 'none', 'markerfacealpha', .5)
hold on
p = polyfit(first_step_all', memory_viewing(i,:)',1);
new_x = linspace(0,1,100);
plot(new_x, polyval(p, new_x), 'color', cmp(i,:), 'linewidth', 1)
set(gca,'ytick', 0:.5:1)
axis square
xlabel('Char. Cond. Prob. movie')
ylabel({'Model activation cond. prob.'})

all_r_cond = [];
all_p_cond = [];
for i = 1:5

[rho_1, p_1] = corr(first_step_all', memory_viewing(i,:)','type', 'spearman');


[rho_2, p_2] = corr(first_step_all', memory_response(i,:)', 'type', 'spearman');
[rho_3, p_3] = corr(second_step_all', memory_response(i,:)', 'type', 'spearman');

all_r_cond = [all_r_cond; rho_1 rho_2 rho_3];
all_p_cond = [all_p_cond; p_1 p_2 p_3];
end

text(.05, 1.15, sprintf('r=%3.2f, p=%3.2e', all_r_cond(i,1), all_p_cond(i,1)))

axes('position', panel_pos(5,:).*[x_pos(3) 1 .35 .9])
[n, x] = hist(memory_response_trimmed(:)- memory_viewing_trimmed(:),-.5:.05:.5);
stairs(x, 100*n/sum(n), 'k', 'linewidth', 1.5)
xlim([-.25 .25])
box off
xlabel({'Model Cond. prob.', 'response - clip'})
ylabel('Associations (%)')
set(gca,'ytick', 0:15:30)
% savefig(fullfile(figures_path, 'Figure_4_July19'), fig, true, capstr, 600);

%%
final_pred_actual = all_patients_prediction;
final_pred_actual(final_pred_actual <.5) = nan;
figure(5)
clf
subplot(221)
hist(final_pred_actual(all_patients_seen & all_patients_charac_in),.0:.05:5)
% bar(1, mean(final_pred(all_patients_seen & all_patients_charac_in)))
% hold on
% plot(1*ones(size(final_pred(all_patients_seen & all_patients_charac_in))), final_pred_actual(all_patients_seen & all_patients_charac_in), '.','color', zgray)

subplot(222)
hist(final_pred_actual(~all_patients_seen & all_patients_charac_in), 0:.05:5)
% bar(2, mean(final_pred(~all_patients_seen & all_patients_charac_in)))
% plot(2*ones(size(final_pred(~all_patients_seen & all_patients_charac_in))), final_pred_actual(~all_patients_seen & all_patients_charac_in), '.','color', zgray)

subplot(223)
hist(final_pred_actual(all_patients_seen & ~all_patients_charac_in), 0:.05:5)
% bar(3, mean(final_pred(all_patients_seen & ~all_patients_charac_in)))
% plot(3*ones(size(final_pred(all_patients_seen & ~all_patients_charac_in))), final_pred_actual(all_patients_seen & ~all_patients_charac_in), '.','color', zgray)

subplot(224)
hist(final_pred_actual(~all_patients_seen & ~all_patients_charac_in),0:.05:5)
% bar(4, mean(final_pred(~all_patients_seen & ~all_patients_charac_in)))
% plot(4*ones(size(final_pred(~all_patients_seen & ~all_patients_charac_in))), final_pred_actual(~all_patients_seen & ~all_patients_charac_in), '.','color', zgray)
%%
ic = 2;
cond1 = all_patients_charac_in == 1 & all_patients_seen == 1 & all_patients_char_number==ic;
cond2 = all_patients_charac_in == 1 & all_patients_seen == 0 & all_patients_char_number==ic;
cond3 = all_patients_charac_in == 0 & all_patients_seen == 1 & all_patients_char_number==ic;
cond4 = all_patients_charac_in == 0 & all_patients_seen == 0 & all_patients_char_number==ic;

clf
colormap(hot)
count = 1;
for i = 1 : 4
    this_count = sum(eval(strcat('cond',num2str(i))));
    pcolor(time_axis, count:count+this_count - 1, ...
        all_patients_prediction_exp(eval(strcat('cond',num2str(i))),:))
    shading interp
    count = count + this_count;
    hold on
%     pause
end
%%
for i = 1 : 4
shadedErrorBar(time_axis, nanmean(all_patients_prediction_exp(eval(strcat('cond',num2str(i))),:)),...
    nanstd(all_patients_prediction_exp(eval(strcat('cond',num2str(i))),:))/...
    sqrt(sum(eval(strcat('cond',num2str(i))))), {'color', cmp(i,:)}, .3)
hold on
end
%%
for ip = 1:4
    figure(ip)
    clf
time_axis =  linspace(-1,1,14);
for iC = 1 : 4
    subplot(2,2,iC)
    this_memory = all_memory_results(ip).start;
    this_memory = reshape(this_memory, [size(this_memory,1)*size(this_memory,2), size(this_memory,3), size(this_memory,4)]);
shadedErrorBar(time_axis,nanmean(squeeze(this_memory(:,iC,:))),...
    nanstd(squeeze(this_memory(:,iC,:)))/sqrt(size(this_memory, 1)), {'color', cmp(iC,:)}, 1)
end
end
%%
for ip = 1:4
    figure(ip)
    clf
time_axis =  linspace(-1,1,14);
for iC = 1 : 4
    subplot(2,2,iC)
    this_memory = all_memory_results(ip).end;
    this_memory = reshape(this_memory, [size(this_memory,1)*size(this_memory,2), size(this_memory,3), size(this_memory,4)]);
    meh2 = nanmean(squeeze(this_memory(:,iC,8:14)),2);
    meh1 = nanmean(squeeze(this_memory(:,iC,1:7)),2);
    hist(meh2-meh1,[-1:.05:1])
    ylim([0 100])
    [p]=ranksum(meh1, meh2)
    pause
end
end
%%
clf
first_step_all = [];
second_step_all = [];
first_spisode_cond_all = [];
second_episode_cond_all =  [];

for ip = 1:4
       figure(ip)
    clf
    data = all_memory_results(ip);
    first_step_all = [first_step_all; first_step_associations(:)];
    second_step_all = [second_step_all; second_step_associations(:)];
    first_spisode_cond_all = [first_spisode_cond_all; data.cond_prob_clip(1,:)'];
    second_episode_cond_all =  [second_episode_cond_all; data.cond_prob_clip(2,:)'];
    
    
    subplot(221)
%     plot(first_step_associations(:), data.cond_prob_clip(2,:),'*r')
    hold on
    plot(first_step_associations(:), data.cond_prob_clip(1,:),'*b')
    [rho, p] = corr(first_step_associations(:), data.cond_prob_clip(1,:)')
    slope = polyfit(first_step_associations(:), data.cond_prob_clip(1,:)', 1)
    title(sprintf('E1: First step \n r = %3.2f and p = %3.2e, slope = %3.2f', rho, p, slope(1)))
    axis square
    box off
    xlim([0 1])
    ylim([0 1])
    ylabel('NN Prediction Conditional Probablity')
    
    subplot(222)
%     plot(first_step_associations(:), data.cond_prob_clip(2,:),'*r')
    hold on
    plot(first_step_associations(:), data.cond_prob_clip(2,:),'*r')
    [rho, p] = corr(first_step_associations(:), data.cond_prob_clip(2,:)')
    slope = polyfit(first_step_associations(:), data.cond_prob_clip(2,:)', 1)
    title(sprintf('E2: First step \n r = %3.2f and p = %3.2e, slope = %3.2f', rho, p, slope(1)))
    axis square
    box off
    xlim([0 1])
    ylim([0 1])
   
    
    subplot(223)
    hold on
    plot(second_step_associations(:), data.cond_prob_clip(1,:),'*b')
    [rho, p] = corr(second_step_associations(:), data.cond_prob_clip(1,:)')
    slope = polyfit(second_step_associations(:), data.cond_prob_clip(1,:)', 1)
    title(sprintf('E1: Second step \n r = %3.2f and p = %3.2e, slope = %3.2f', rho, p, slope(1)))
    axis square
    box off
    xlim([0 1])
    ylim([0 1])
    suptitle(strcat('Patient Number: ', num2str(ip)))
    xlabel('Movie Conditional Probablity')
    ylabel('NN Prediction Conditional Probablity')
    
     subplot(224)
    hold on
    plot(second_step_associations(:), data.cond_prob_clip(2,:),'*r')
    [rho, p] = corr(second_step_associations(:), data.cond_prob_clip(2,:)')
    slope = polyfit(second_step_associations(:), data.cond_prob_clip(2,:)', 1)
    title(sprintf('E2: Second step \n r = %3.2f and p = %3.2e, slope = %3.2f', rho, p, slope(1)))
    axis square
    box off
    xlim([0 1])
    ylim([0 1])
    suptitle(strcat('Patient Number: ', num2str(ip)))
    xlabel('Movie Conditional Probablity')
    saveas(gca,fullfile(pwd,strcat('clip_cond_prob_pt', num2str(ip), '.png')))
end
%% Neural firing ratemap (part of figure 2)
colormap(hot)
tBins = units(1).Firing.Centers;
this_firing = arrayfun(@(x) x.Firing.Rate, units, 'uniformoutput', false);
this_firing = cell2mat(this_firing')';
pcolor(tBins, 1:size(this_firing,1), this_firing); shading flat
%% supplementary figure 1: data : LSTM: overall accuracy of the model - loading data from patients





patientNums = {'431', '433', '435', '436', '439', '441', '442', '444', '445', '452'}; 
this_model  = model_types{3};
all_kfold_results = [];
for iP = 1:length(patientNums)  
    iP
    try
    this_patient_kfold = fullfile(path_to_processed_data, this_model, patientNums{iP})    
    load(fullfile(this_patient_kfold, strcat('kfold_results_fig1_', patientNums{iP}, '.mat')))
    all_kfold_results = [all_kfold_results; data];
    end
end
    
    
    
    
    
    
    
