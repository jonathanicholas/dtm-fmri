%
% Create Documents via Symbolic Aggregate approXimation (SAX)
% Output is text analogue of a functional connectivity matrix
%
% Make sure you change the timeseries that are loaded
% As well as the text directory output
%
function runSAX_WM_RL(d)

    addpath('SAX_dependencies/')

    load('WM_RL_timeseries.mat')
    ts_dims = size(ts);

    nROIs = ts_dims(2);
    nSubjects = ts_dims(1);

    symbols = {'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o'};

    subs = (1:nSubjects);

    alphabet_size = 2;

    word_windows = [];
    doc = [];

    data = squeeze(ts(d,:,:));
    data_len = ts_dims(3);
    nseg = data_len;
    sax = zeros(nROIs,nseg);

    sax_string = {};
    for i = 1:nROIs
        str = timeseries2symbol(data(i,:), data_len, nseg, alphabet_size, 2, data(i,:));
        sax(i,:) = str;
        sax_string(i,:) = symbols(str);
    end

    sax_string = sax_string';

    words = [];
    for i = 1:nseg
        for j = 1:nROIs
            for k = j+1:nROIs
                if k ~= j
                    word = [sax_string(i,j),sax_string(i,k),num2str(k),'_',num2str(j)];
                    word = strcat(word{1:length(word)});
                    words = [words ' ' word];
                end
            end
        end
    end
    word_windows = [word_windows words];

    fid = fopen(['../DTM/texts/WM_RL/subject',int2str(d),'_dim',num2str(alphabet_size),'.txt'], 'w');
    fprintf(fid, '%s', (word_windows));
    fclose(fid);
