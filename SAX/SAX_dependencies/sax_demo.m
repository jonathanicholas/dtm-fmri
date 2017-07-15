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
% This code provides a step-by-step demo of SAX (Symbolic Aggregate approXimation).  Press enter
% for the next step.
% 
%   usage: [str] = sax_demo
%          [str] = sax_demo(data)
%
% Copyright (c) 2003, Eamonn Keogh, Jessica Lin, Stefano Lonardi, Pranav Patel, Li Wei. All rights reserved.
%
function [sax_string] = sax_demo(data)

    if nargin == 0
        data_len      = 256;
        data = random_walk(data_len);
    else
        data_len      = length(data);
    end
    
    nseg          = 8;
    alphabet_size = 4;

    if alphabet_size > 10
        disp('Currently alphabet_size cannot be larger than 10.  Please update the breakpoint table if you wish to do so');
        return;
    end
    
    data_len
    nseg
    
    % nseg must be divisible by data length
    if (mod(data_len, nseg))
        
        disp('nseg must be divisible by the data length. Aborting ');
        return;  
        
    end;

    % win_size is the number of data points on the raw time series that will be mapped to a 
    % single symbol
    win_size = floor(data_len/nseg)
    
    data = (data - mean(data))/std(data);

    plot(data);

    pause;
    
    % special case: no dimensionality reduction
    if data_len == nseg
        PAA = data;
        
    % Convert to PAA.  Note that this line is also in timeseries2symbol, which will be
    % called later.  So it's redundant here and is for the purpose of plotting only.
    else
        PAA = [mean(reshape(data,win_size,nseg))];                     
    end
    
    % plot the PAA segments
    PAA_plot = repmat(PAA', 1, win_size);
    PAA_plot = reshape(PAA_plot', 1, data_len)';
    
    hold on;
    plot(PAA_plot,'r');
    
    pause;

    % map the segments to string
    str = timeseries2symbol(data, data_len, nseg, alphabet_size);
    
    % get the breakpoints
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
    guidelines = repmat(cutlines', 1, data_len);    
    plot(guidelines', 'color', [0.8 0.8 0.8]);
    hold on    
    
    pause;
    
    color = {'g', 'y', 'm', 'c'};
    symbols = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'};
    
    % high-light the segments and assign them to symbols
    for i = 1 : nseg
        
        % get the x coordinates for the segments
        x_start = (i-1) * win_size + 1;
        x_end   = x_start + win_size - 1;
        x_mid   = x_start + (x_end - x_start) / 2;

        % color-code each segment
        colorIndex = rem(str(i),length(color))+1;
        
        % draw the segments
        plot([x_start:x_end],PAA_plot([x_start:x_end]), 'color', color{colorIndex}, 'linewidth', 3);

        % show symbols
        text(x_mid, PAA_plot(x_start), symbols{str(i)}, 'fontsize', 14);
    end
    
    sax_string = symbols(str);
%end


%------------------------------------------------------------------------------------------
% Make random walk data
%------------------------------------------------------------------------------------------

function r = random_walk(n)
% r = random_walk(n)
% n: length of random walk time series
% 
% This is the continuous analog of symmetric random walk, each increment y(s+t)-y(s) is 
% Gaussian with distribution N(0,t^2) and increments over disjoint intervals are independent. 
% It is typically simulated as an approximating random walk in discrete time. 

sigma=1;
r=[0 cumsum(sigma.*randn(1,n-1))]; % standard Brownian motion 