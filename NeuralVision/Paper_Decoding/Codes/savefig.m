function savefig(fname,  fig, skipeps, capstr, dpi)
    if nargin < 5
        dpi = 600;
    end
    if nargin < 4
        capstr = '';
    end
    if nargin < 3
        skipeps = false;
    end
    if nargin < 2
        fig = gcf();
    end
    
    figure(gcf);
    
    set(fig, 'units', get(fig, 'paperunits'), 'position', [1,2, get(fig, 'papersize')]);
    if ~skipeps
        print(gcf, fname, '-depsc2', '-cmyk', '-painters');
%        export_fig(strcat(fname, '_cmyk'), '-tif', '-eps', '-r600', '-cmyk', '-nocrop')
    end
    
    % print rgb tiff
    print(gcf, fname, strcat('-r', num2str(dpi)), '-dtiff');    
    
    % and make cmyk version
    
    %{
    rgb = imread(strcat(fname, '.tif'));
    cform = makecform('srgb2cmyk');
    lab = applycform(rgb, cform);
    imwrite(lab, strcat(fname, '_cmyk.tif'));
    %}

    
    if ~strcmp(capstr, '')
        f = fopen(strcat(fname, '_caption.txt'), 'w');
        fwrite(f, capstr);
        fclose(f);
    end
    
    % tikz! ftw!
%     matlab2tikz(strcat(fname, '.tikz'), 'figurehandle', fig, 'imagesAsPng', false, 'height', '\figureheight', 'width', '\figurewidth');
    
    %print(gcf, strcat(fname, '_rgb'), '-r300', '-dpng');
    %export_fig(strcat(fname, '_rgb'), '-tif', '-r600', '-rgb', '-nocrop')

    %close(fig)    
    
    %export_fig(strcat('F:\Bernard\Documents\Dropbox\First Fifteen Paper\Figures\', fname, '.png'));
    %if ~skipeps
    %    export_fig(strcat('F:\Bernard\Documents\Dropbox\First Fifteen Paper\Figures\', fname, '.eps'), '-cmyk');
    %end
    %saveas(gcf, strcat('F:\Bernard\Documents\Dropbox\First Fifteen Paper\Figures\', fname, '.fig'));
    
    