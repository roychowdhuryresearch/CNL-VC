%% Define paths
close all
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
addpath(genpath('./Codes'))
figures_path = './Figures1';
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

load('/media/yipeng/data/movie/Movie_Analysis/fig1.mat')

 
if exist('fig', 'var'); try close(fig); end; end %#ok<TRYNC>
[panel_pos, fig, all_ax] = panelFigureSetup2(5, 3, ['a    b    c    '], ...
    figure_width_triple_col, figure_width_double_col*.75,1, 1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
axes('position', panel_pos(1,:).*[1 .95 6 1.3])


nn_structure = imread(fullfile(figures_path, 'fig1_1.png'));
%imshow(nn_structure,'InitialMagnification', 'fit')
imshow(nn_structure)
axis tight

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
%axes('position', panel_pos(6,:).*[-1 0.9 8.05 1.5])
axes('position', panel_pos(6,:).*[0.1 0.9 6.5 1.3])
%axes('position', panel_pos(6,:).*[1 1 6.5 1.5])

nn_structure = imread(fullfile(figures_path, 'fig1_6.jpeg'));
%imshow(nn_structure,'InitialMagnification', 'fit')
imshow(nn_structure)
axis tight

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
confusion_cmp = colormap(othercolor('Blues7'));
colormap(confusion_cmp)
confusion_1 = confusion.*0;
confusion_1(1,:,:) = confusion(5,:,:);
confusion_1(2:5,:,:) = confusion(1:4,:,:);
for char_num = 1 : 5
axes('position', panel_pos(10+char_num,:).*[0.95 0.9 0.9 0.9])
this_matrix = (squeeze(confusion_1(char_num,1:2,1:2)));
this_matrix = 100 * this_matrix ./ sum(this_matrix, 2)
h = heatmap(this_matrix(1:2, 1:2))
if char_num == 1
h.Title = "Overall.";
else
h.Title = strcat('C. ', num2str(char_num-1) );
end
if char_num ==1
h.YDisplayLabels = {'Y', 'N'};
else
h.YDisplayLabels = {'', ''};
end
h.XDisplayLabels = {'Y_p', 'N_p'};
% box off
if char_num ~=5
h.ColorbarVisible = 'off'
end
set(gca,'Fontsize',11);
end

% freezeColors()
% overall character accuracy
pt_colormap = othercolor('BrBG10', 10)
capstr = [];
savefig(fullfile(figures_path, 'Figure_1_Aug14.tif'), fig, true, capstr, 600);