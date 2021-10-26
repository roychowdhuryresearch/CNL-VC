function all_knockout_results= fixMovieRegionNames(all_knockout_results)
%     all_regions =  arrayfun(@(x) x.final_unique_regions', all_knockout_results, 'uniformoutput', false);
%     all_regions = cat(1,all_regions{:});
%     all_microwires =  arrayfun(@(x) x.final_microwire_regions', all_knockout_results, 'uniformoutput', false);
%     all_microwires = cat(1, all_microwires{:});
%     all_patients_regions = cell2mat(arrayfun(@(x) x.patientNum_regions, all_knockout_results, 'uniformoutput', false));
%     all_patients_microwires = cell2mat(arrayfun(@(x) x.patientNum_microwires, all_knockout_results, 'uniformoutput', false));
%     

    %------------- fixing 431
    inds = find(strcmp(all_knockout_results(1).final_unique_regions, 'LIP'));
    if ~isempty(inds)
    for ii = 1 :  length(inds)
        i = inds(ii);
        all_knockout_results(1).final_unique_regions{i} = 'LPC';
    end
    end

    sig_inds = find(strcmp(all_knockout_results(1).final_microwire_regions, 'LIP'));
    if ~isempty(sig_inds)
    for jj = 1 : length(sig_inds)
        j = sig_inds(jj);
        all_knockout_results(1).final_microwire_regions{j} = 'LPC';
    end
    end

    %--
    inds = find(strcmp(all_knockout_results(1).final_unique_regions, 'RSS'));
    if ~isempty(inds)
    for ii = 1 :  length(inds)
        i = inds(ii);
        all_knockout_results(1).final_unique_regions{i} = 'RMC';
    end
    end

    sig_inds = find(strcmp(all_knockout_results(1).final_microwire_regions, 'RSS'));
    if ~isempty(sig_inds)
    for jj = 1 : length(sig_inds)
        j = sig_inds(jj);
        all_knockout_results(1).final_microwire_regions{j} = 'RMC';
    end
    end

    %--
    inds = find(strcmp(all_knockout_results(1).final_unique_regions, 'RIPA'));
    if ~isempty(inds)
    for ii = 1 :  length(inds)
        i = inds(ii);
        all_knockout_results(1).final_unique_regions{i} = 'RPC';
    end
    end

    sig_inds = find(strcmp(all_knockout_results(1).final_microwire_regions, 'RIPA'));
    if ~isempty(sig_inds)
    for jj = 1 : length(sig_inds)
        j = sig_inds(jj);
        all_knockout_results(1).final_microwire_regions{j} = 'RPC';
    end
    end

    %--
    inds = find(strcmp(all_knockout_results(1).final_unique_regions, 'RIPP'));
    if ~isempty(inds)
    for ii = 1 :  length(inds)
        i = inds(ii);
        all_knockout_results(1).final_unique_regions{i} = 'RPC';
    end
    end

    sig_inds = find(strcmp(all_knockout_results(1).final_microwire_regions, 'RIPP'));
    if ~isempty(sig_inds)
    for jj = 1 : length(sig_inds)
        j = sig_inds(jj);
        all_knockout_results(1).final_microwire_regions{j} = 'RPC';
    end
    end
    %---------------- fixing 433
    inds = find(strcmp(all_knockout_results(2).final_unique_regions, 'LOF'));
    if ~isempty(inds)
    for ii = 1 :  length(inds)
        i = inds(ii);
        all_knockout_results(2).final_unique_regions{i} = 'LVMPFC';
    end
    end

    sig_inds = find(strcmp(all_knockout_results(2).final_microwire_regions, 'LOF'));
    if ~isempty(sig_inds)
    for jj = 1 : length(sig_inds)
        j = sig_inds(jj);
        all_knockout_results(2).final_microwire_regions{j}  = 'LVMPFC';
    end
    end
    %--
    inds = find(strcmp(all_knockout_results(2).final_unique_regions, 'LPH'));
    if ~isempty(inds)
    for ii = 1 :  length(inds)
        i = inds(ii);
        all_knockout_results(2).final_unique_regions{i} = 'LPHG';
    end
    end

    sig_inds = find(strcmp(all_knockout_results(2).final_microwire_regions, 'LPH'));
    if ~isempty(sig_inds)
    for jj = 1 : length(sig_inds)
        j = sig_inds(jj);
        all_knockout_results(2).final_microwire_regions{j}  = 'LPHG';
    end
    end
    %--
    inds = find(strcmp(all_knockout_results(2).final_unique_regions, 'RPH'));
    if ~isempty(inds)
    for ii = 1 :  length(inds)
        i = inds(ii);
        all_knockout_results(2).final_unique_regions{i} = 'RSUB';
    end
    end

    sig_inds = find(strcmp(all_knockout_results(2).final_microwire_regions, 'RPH'));
    if ~isempty(sig_inds)
    for jj = 1 : length(sig_inds)
        j = sig_inds(jj);
        all_knockout_results(2).final_microwire_regions{j}  = 'RSUB';
    end
    end

    %---------------- fixing 435
    inds = find(strcmp(all_knockout_results(3).final_unique_regions, 'LAH'));
    if ~isempty(inds)
    for ii = 1 :  length(inds)
        i = inds(ii);
        all_knockout_results(3).final_unique_regions{i} = 'LSUB';
    end
    end

    sig_inds = find(strcmp(all_knockout_results(3).final_microwire_regions, 'LAH'));
    if ~isempty(sig_inds)
    for jj = 1 : length(sig_inds)
        j = sig_inds(jj);
        all_knockout_results(3).final_microwire_regions{j}  = 'LSUB';
    end
    end

    %---------------- fixing 436
    inds = find(strcmp(all_knockout_results(4).final_unique_regions, 'LOF'));
    if ~isempty(inds)
    for ii = 1 :  length(inds)
        i = inds(ii);
        all_knockout_results(4).final_unique_regions{i} = 'LVMPFC';
    end
    end

    sig_inds = find(strcmp(all_knockout_results(4).final_microwire_regions, 'LOF'));
    if ~isempty(sig_inds)
    for jj = 1 : length(sig_inds)
        j = sig_inds(jj);
        all_knockout_results(4).final_microwire_regions{j}  = 'LVMPFC';
    end
    end

    %--
    inds = find(strcmp(all_knockout_results(4).final_unique_regions, 'ROF'));
    if ~isempty(inds)
    for ii = 1 :  length(inds)
        i = inds(ii);
        all_knockout_results(4).final_unique_regions{i} = 'RVMPFC';
    end
    end

    sig_inds = find(strcmp(all_knockout_results(4).final_microwire_regions, 'ROF'));
    if ~isempty(sig_inds)
    for jj = 1 : length(sig_inds)
        j = sig_inds(jj);
        all_knockout_results(4).final_microwire_regions{j}  = 'RVMPFC';
    end
    end

    %--
    inds = find(strcmp(all_knockout_results(4).final_unique_regions, 'RMH'));
    if ~isempty(inds)
    for ii = 1 :  length(inds)
        i = inds(ii);
        all_knockout_results(4).final_unique_regions{i} = 'RMH-EC';
    end
    end

    sig_inds = find(strcmp(all_knockout_results(4).final_microwire_regions, 'RMH'));
    if ~isempty(sig_inds)
    for jj = 1 : length(sig_inds)
        j = sig_inds(jj);
        all_knockout_results(4).final_microwire_regions{j}  = 'RMH-EC';
    end
    end
    %---------------- fixing 439
    inds = find(strcmp(all_knockout_results(5).final_unique_regions, 'LOF'));
    if ~isempty(inds)
    for ii = 1 :  length(inds)
        i = inds(ii);
        all_knockout_results(5).final_unique_regions{i} = 'LVMPFC';
    end
    end

    sig_inds = find(strcmp(all_knockout_results(5).final_microwire_regions, 'LOF'));
    if ~isempty(sig_inds)
    for jj = 1 : length(sig_inds)
        j = sig_inds(jj);
        all_knockout_results(5).final_microwire_regions{j}  = 'LVMPFC';
    end
    end
    %--
    inds = find(strcmp(all_knockout_results(5).final_unique_regions, 'ROF'));
    if ~isempty(inds)
    for ii = 1 :  length(inds)
        i = inds(ii);
        all_knockout_results(5).final_unique_regions{i} = 'RVMPFC';
    end
    end

    sig_inds = find(strcmp(all_knockout_results(5).final_microwire_regions, 'ROF'));
    if ~isempty(sig_inds)
    for jj = 1 : length(sig_inds)
        j = sig_inds(jj);
        all_knockout_results(5).final_microwire_regions{j}  = 'RVMPFC';
    end
    end
    %--
    inds = find(strcmp(all_knockout_results(5).final_unique_regions, 'LMH'));
    if ~isempty(inds)
    for ii = 1 :  length(inds)
        i = inds(ii);
        all_knockout_results(5).final_unique_regions{i} = 'LSUB';
    end
    end

    sig_inds = find(strcmp(all_knockout_results(5).final_microwire_regions, 'LMH'));
    if ~isempty(sig_inds)
    for jj = 1 : length(sig_inds)
        j = sig_inds(jj);
        all_knockout_results(5).final_microwire_regions{j}  = 'LSUB';
    end
    end
    %---------------- fixing 441
    inds = find(strcmp(all_knockout_results(6).final_unique_regions, 'LOF'));
    if ~isempty(inds)
    for ii = 1 :  length(inds)
        i = inds(ii);
        all_knockout_results(6).final_unique_regions{i} = 'LVMPFC';
    end
    end

    sig_inds = find(strcmp(all_knockout_results(6).final_microwire_regions, 'LOF'));
    if ~isempty(sig_inds)
    for jj = 1 : length(sig_inds)
        j = sig_inds(jj);
        all_knockout_results(6).final_microwire_regions{j}  = 'LVMPFC';
    end
    end

    %--
    inds = find(strcmp(all_knockout_results(6).final_unique_regions, 'ROF'));
    if ~isempty(inds)
    for ii = 1 :  length(inds)
        i = inds(ii);
        all_knockout_results(6).final_unique_regions{i} = 'RAC';
    end
    end

    sig_inds = find(strcmp(all_knockout_results(6).final_microwire_regions, 'ROF'));
    if ~isempty(sig_inds)
    for jj = 1 : length(sig_inds)
        j = sig_inds(jj);
        all_knockout_results(6).final_microwire_regions{j}  = 'RAC';
    end
    end
    %---------------- fixing 442
%     inds = find(strcmp(all_knockout_results(7).final_unique_regions, 'LOF'));
%     if ~isempty(inds)
%     for ii = 1 :  length(inds)
%         i = inds(ii);
%         all_knockout_results(7).final_unique_regions{i} = 'LVMPFC';
%     end
%     end
% 
%     sig_inds = find(strcmp(all_knockout_results(7).final_microwire_regions, 'LOF'));
%     if ~isempty(sig_inds)
%     for jj = 1 : length(sig_inds)
%         j = sig_inds(jj);
%         all_knockout_results(7).final_microwire_regions{j}  = 'LVMPFC';
%     end
%     end
%     %--
%     inds = find(strcmp(all_knockout_results(7).final_unique_regions, 'ROF'));
%     if ~isempty(inds)
%     for ii = 1 :  length(inds)
%         i = inds(ii);
%         all_knockout_results(7).final_unique_regions{i} = 'RVMPFC';
%     end
%     end
% 
%     sig_inds = find(strcmp(all_knockout_results(7).final_microwire_regions, 'ROF'));
%     if ~isempty(sig_inds)
%     for jj = 1 : length(sig_inds)
%         j = sig_inds(jj);
%         all_knockout_results(7).final_microwire_regions{j}  = 'RVMPFC';
%     end
%     end

    %---------------- fixing 444
    inds = find(strcmp(all_knockout_results(7).final_unique_regions, 'LOF'));
    if ~isempty(inds)
    for ii = 1 :  length(inds)
        i = inds(ii);
        all_knockout_results(7).final_unique_regions{i} = 'LAC';
    end
    end

    sig_inds = find(strcmp(all_knockout_results(7).final_microwire_regions, 'LOF'));
    if ~isempty(sig_inds)
    for jj = 1 : length(sig_inds)
        j = sig_inds(jj);
        all_knockout_results(7).final_microwire_regions{j}  = 'LAC';
    end
    end
    %--
    inds = find(strcmp(all_knockout_results(7).final_unique_regions, 'ROF'));
    if ~isempty(inds)
    for ii = 1 :  length(inds)
        i = inds(ii);
        all_knockout_results(7).final_unique_regions{i} = 'RVMPFC';
    end
    end

    sig_inds = find(strcmp(all_knockout_results(7).final_microwire_regions, 'ROF'));
    if ~isempty(sig_inds)
    for jj = 1 : length(sig_inds)
        j = sig_inds(jj);
        all_knockout_results(7).final_microwire_regions{j}  = 'RVMPFC';
    end
    end
    %---------------- fixing 445
    inds = find(strcmp(all_knockout_results(8).final_unique_regions, 'LOF'));
    if ~isempty(inds)
    for ii = 1 :  length(inds)
        i = inds(ii);
        all_knockout_results(8).final_unique_regions{i} = 'LVMPFC';
    end
    end

    sig_inds = find(strcmp(all_knockout_results(8).final_microwire_regions, 'LOF'));
    if ~isempty(sig_inds)
    for jj = 1 : length(sig_inds)
        j = sig_inds(jj);
        all_knockout_results(8).final_microwire_regions{j}  = 'LVMPFC';
    end
    end
    %--
    inds = find(strcmp(all_knockout_results(8).final_unique_regions, 'ROF'));
    if ~isempty(inds)
    for ii = 1 :  length(inds)
        i = inds(ii);
        all_knockout_results(8).final_unique_regions{i} = 'RVMPFC';
    end
    end

    sig_inds = find(strcmp(all_knockout_results(8).final_microwire_regions, 'ROF'));
    if ~isempty(sig_inds)
    for jj = 1 : length(sig_inds)
        j = sig_inds(jj);
        all_knockout_results(8).final_microwire_regions{j}  = 'RVMPFC';
    end
    end
    %---------------- fixing 452
    inds = find(strcmp(all_knockout_results(9).final_unique_regions, 'LOF'));
    if ~isempty(inds)
    for ii = 1 :  length(inds)
        i = inds(ii);
        all_knockout_results(9).final_unique_regions{i} = 'LVMPFC';
    end
    end

    sig_inds = find(strcmp(all_knockout_results(9).final_microwire_regions, 'LOF'));
    if ~isempty(sig_inds)
    for jj = 1 : length(sig_inds)
        j = sig_inds(jj);
        all_knockout_results(9).final_microwire_regions{j}  = 'LVMPFC';
    end
    end
    %--
    inds = find(strcmp(all_knockout_results(9).final_unique_regions, 'ROF'));
    if ~isempty(inds)
    for ii = 1 :  length(inds)
        i = inds(ii);
        all_knockout_results(9).final_unique_regions{i} = 'RVMPFC';
    end
    end

    sig_inds = find(strcmp(all_knockout_results(9).final_microwire_regions, 'ROF'));
    if ~isempty(sig_inds)
    for jj = 1 : length(sig_inds)
        j = sig_inds(jj);
        all_knockout_results(9).final_microwire_regions{j}  = 'RVMPFC';
    end
    end
    %--
    inds = find(strcmp(all_knockout_results(9).final_unique_regions, 'LAH'));
    if ~isempty(inds)
    for ii = 1 :  length(inds)
        i = inds(ii);
        all_knockout_results(9).final_unique_regions{i} = 'LSUB';
    end
    end

    sig_inds = find(strcmp(all_knockout_results(9).final_microwire_regions, 'LAH'));
    if ~isempty(sig_inds)
    for jj = 1 : length(sig_inds)
        j = sig_inds(jj);
        all_knockout_results(9).final_microwire_regions{j}  = 'LSUB';
    end
    end
    %------
    inds = find(strcmp(all_knockout_results(9).final_unique_regions, 'RMH'));
    if ~isempty(inds)
    for ii = 1 :  length(inds)
        i = inds(ii);
        all_knockout_results(9).final_unique_regions{i} = 'RSUB';
    end
    end

    sig_inds = find(strcmp(all_knockout_results(9).final_microwire_regions, 'RMH'));
    if ~isempty(sig_inds)
    for jj = 1 : length(sig_inds)
        j = sig_inds(jj);
        all_knockout_results(9).final_microwire_regions{j}  = 'RSUB';
    end
    end

    %--
    inds = find(strcmp(all_knockout_results(9).final_unique_regions, 'RTP'));
    if ~isempty(inds)
    for ii = 1 :  length(inds)
        i = inds(ii);
        all_knockout_results(9).final_unique_regions{i} = 'RO';
    end
    end

    sig_inds = find(strcmp(all_knockout_results(9).final_microwire_regions, 'RTP'));
    if ~isempty(sig_inds)
    for jj = 1 : length(sig_inds)
        j = sig_inds(jj);
        all_knockout_results(9).final_microwire_regions{j}  = 'RO';
    end
    end
end