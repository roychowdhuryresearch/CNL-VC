function [ panel_pos, fig, all_ax ] = panelFigureSetup2( panel_x, panel_y, labels, figure_width, ...
    figure_height, panel_spacing_paper, panel_margin_paper )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

if nargin < 7
    panel_margin_paper = 1.5; % cm
end
if nargin < 6
    panel_spacing_paper = 2.0; % cm
end
if nargin < 4
    figure_width = 18.3; % cm  - nature neuro double column
end

%figure_width = figure_width_double_col;
%panel_spacing_paper = 2.0; % cm
%panel_margin_paper = 1.5; % cm
panel_spacing_x = panel_spacing_paper / figure_width;
panel_margin_x = panel_margin_paper / figure_width;
%5panel_x = 3;
%panel_y = 2;
panel_width = (1 - (panel_x - 1) * panel_spacing_x - 2 * panel_margin_x) / panel_x;
%panel_width_paper = figure_width * panel_width;
%figure_height = panel_width_paper * panel_y + panel_spacing_paper * (panel_y - 1) + 2 * panel_margin_paper;
panel_spacing_y = panel_spacing_paper / figure_height;
panel_margin_y = panel_margin_paper / figure_height;
%panel_height = panel_width_paper / figure_height;
panel_height = (1 - (panel_y - 1) * panel_spacing_y - 2 * panel_margin_y) / panel_y;

fig = figure('color', 'w', 'paperunits', 'centimeters', 'papersize', [figure_width, figure_height], ...
    'paperposition',  [0, 0, figure_width, figure_height]);
set(fig, 'units', get(gcf, 'paperunits'), 'position', [1,5, get(gcf, 'papersize')]);
all_ax = axes('position', [0,0,1,1], 'visible', 'off');
%plot_axes = zeros(panel_x * panel_y,1);

panel_pos = zeros(panel_x * panel_y,4);
for i = 1 : panel_x * panel_y
    rws = floor((i-1)/panel_x);
    panel_pos(i,:) = [panel_margin_x + mod(i-1, panel_x) * (panel_spacing_x + panel_width), ...
                    1 - panel_margin_y - (rws+1) * panel_height - rws*panel_spacing_y,...
                    panel_width, panel_height];
end

% Now add the panel labels
%labels=['ACEBDF'];
for i = 1 : panel_x * panel_y
    if labels(i) == ' '
        continue
    end
    h = text(panel_pos(i,1)-panel_spacing_x/4, panel_pos(i,2) + panel_pos(i,4) + panel_spacing_y/8, labels(i), 'fontweight', 'bold'); %%
    %h = text(panel_spacing_x /2 + mod(i-1, panel_x) * (panel_spacing_x + panel_width), ...
    %    1 - (1 + floor((i-1)/panel_x)) * (panel_spacing_y + panel_height) + panel_height, ...
    %    labels(i), 'fontweight', 'bold');
    set(h, 'verticalAlignment', 'bottom');
    set(h, 'horizontalAlignment', 'right');
    set(h, 'fontsize', 9);
end

end

