% Copyright and terms of use (DO NOT REMOVE):
% The code is made freely available for non-commercial uses only, provided that the copyright 
% header in each file not be removed, and suitable citation(s) (see below) be made for papers 
% published based on the code.
%
% The code is not optimized for speed, and we are not responsible for any errors that might
% occur in the code.
%
% The copyright of the code is retained by the authors.  By downloading/using this code you
% agree to all the terms stated above.
%
%   Lin, J., Keogh, E., Lonardi, S. & Chiu, B. 
%   "A Symbolic Representation of Time Series, with Implications for Streaming Algorithms." 
%   In proceedings of the 8th ACM SIGMOD Workshop on Research Issues in Data Mining and 
%   Knowledge Discovery. San Diego, CA. June 13, 2003. 
%
%
%   Lin, J., Keogh, E., Patel, P. & Lonardi, S. 
%   "Finding Motifs in Time Series". In proceedings of the 2nd Workshop on Temporal Data Mining, 
%   at the 8th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. 
%   Edmonton, Alberta, Canada. July 23-26, 2002
%
%
%
% Copyright (c) 2003, Eamonn Keogh, Jessica Lin, Stefano Lonardi, Pranav Patel, Li Wei. All rights reserved.
%
% This demo presents a visual comparison between SAX and PAA, and shows how SAX can represent
% data in finer granularity while using the same, if not less, amount of space as PAA.
% 
% The input parameter [data] is optional.  The default # of PAA segments is 16, and the alphabet
% size is 4.
% 

function [] = symbolic_visual(data)

    if nargin == 0
        temp = sin(0:0.2:100)';                  % make a long sine wave
        len = 128;
        data = temp([1:len]);
    end
    
    alphabet_size = 4;
    
    n = 16;
    
    len = length(data);
    
    cmpr_rate = len/n;
    
    % normalize the time series first
    nData = (data - mean(data)) ./ std(data);                           
    
    % if we represent each symbol as a binary string, calculate how much space 
    % (i.e. how many bits) is needed to store one symbol
    num_bits_per_symbol = floor(log2(alphabet_size)+1)

    % 64 bits vs. 3 bits (for alphabet size between 4 and 7) per PAA coef
    PAA_symbolic_ratio = floor(64 / num_bits_per_symbol)          
    
    % this is the maximum symbolic segments possible, using the same amount of space as PAA
    % (assuming each PAA segment uses 8 bytes)
    temp1 = PAA_symbolic_ratio * n                        

    if temp1 >= len
        sym_seg = len;
    
    % can't just use the maximum # of segments possible -- it has to be divisible by the 
    % original length
    else
        temp2 = floor(log2(temp1));    
        sym_seg = 2 .^ temp2;
    end  

    if sym_seg > 4 * n
        sym_seg = sym_seg / 4;
    end    

    %-----------------------------------------------------------------------------
    % plot symbolic
    %-----------------------------------------------------------------------------
    
    subplot(3,1,1);
    str2 = timeseries2symbol(nData', len, n, alphabet_size);
    plot_symbolic(nData, str2, len, n, alphabet_size);    
    title('SAX (# symbols = # PAA segments)');
    
    subplot(3,1,2);
    str = timeseries2symbol(nData', len, sym_seg, alphabet_size);
    plot_symbolic(nData, str, len, sym_seg, alphabet_size);
    title('SAX - finer granularity (using no more space than PAA, have as many symbols as possible)');    
    
    %--------------------------------------------------------------------------
    % plot PAA
    %--------------------------------------------------------------------------
    
    subplot(3,1,3);
    
    seg_size = cmpr_rate;
    
    PAA = [mean(reshape(nData,seg_size, n))];
    PAA = repmat(PAA, seg_size, 1);
    p = reshape(PAA, len, 1);
        
    plot(nData);
    hold on
    plot(p,'r');    
    axis([1 len -3 3])
    
    title('PAA segments');
    
    % make the figure bigger so it's easier to see
    screen=get(0,'screensize');
    set(gcf,'Units','normalized','Position',[0.02  0.0467  0.95  0.85])

function plot_symbolic(data, str, len, sym_seg, alphabet_size)

    seg_size = len / sym_seg;
    
    switch alphabet_size
        case 2, cutlines  = [0];
        case 3, cutlines  = [-0.43 0.43];
        case 4, cutlines  = [-0.67 0 0.67];
        case 5, cutlines  = [-0.84 -0.25 0.25 0.84];
        case 6, cutlines  = [-0.97 -0.43 0 0.43 0.97];
        case 7, cutlines  = [-1.07 -0.57 -0.18 0.18 0.57 1.07];
        case 8, cutlines  = [-1.15 -0.67 -0.32 0 0.32 0.67 1.15];
        case 9, cutlines  = [-1.22 -0.76 -0.43 -0.14 0.14 0.43 0.76 1.22];
        case 10, cutlines = [-1.28 -0.84 -0.52 -0.25 0. 0.25 0.52 0.84 1.28];
        otherwise, disp('WARNING:: Alphabet size too big');
    end;

    % draw the gray guide lines in the background
    guidelines = repmat(cutlines', 1, len);    
    plot(guidelines', 'color', [0.8 0.8 0.8]);
    hold on
    
    max_value = cutlines(alphabet_size-1) + ((cutlines(alphabet_size-1) - cutlines(alphabet_size-2)) * 2);
    min_value = cutlines(1) - ((cutlines(2)-cutlines(1)) * 2);
    
    % include the lower/upper bounds in the cutlines
    % (needed for height calculation)
    new_cutlines = [min_value cutlines max_value];
    
    cutlines_rep = repmat(new_cutlines, sym_seg, 1);
        
    % determine the x position of the rectangles
    x_pos = [1:seg_size:len];

    % determine the y position of the rectangles
    y_pos = cutlines_rep(:,str);

    % determine the (region) height for each rectangle
    heights = cutlines_rep(:,str+1) - y_pos;
    
    
    % draw rectangles
    for i = 1 : sym_seg
        X = [x_pos(1,i) x_pos(1,i)+seg_size x_pos(1,i)+seg_size x_pos(1,i) x_pos(1,i)];
        Y = [y_pos(1,i) y_pos(1,i) y_pos(1,i)+heights(1,i) y_pos(1,i)+heights(1,i) y_pos(1,i)];
        patch(X,Y,'r');
    end

    % draw the actual time series (normalized)
    plot(data,'b.-');

    axis([1 len -3 3])
