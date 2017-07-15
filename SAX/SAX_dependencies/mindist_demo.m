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
%   [1] Lin, J., Keogh, E., Lonardi, S. & Chiu, B. 
%   "A Symbolic Representation of Time Series, with Implications for Streaming Algorithms." 
%   In proceedings of the 8th ACM SIGMOD Workshop on Research Issues in Data Mining and 
%   Knowledge Discovery. San Diego, CA. June 13, 2003. 
%
%
%   [2] Lin, J., Keogh, E., Patel, P. & Lonardi, S. 
%   "Finding Motifs in Time Series". In proceedings of the 2nd Workshop on Temporal Data Mining, 
%   at the 8th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. 
%   Edmonton, Alberta, Canada. July 23-26, 2002
%
% This function demonstrates that mindist lower-bounds the true euclidean distance
%
% Copyright (c) 2003, Eamonn Keogh, Jessica Lin, Stefano Lonardi, Pranav Patel, Li Wei.  All rights reserved.
%
function mindist_demo

temp = sin(0:0.32:20)';                  % make a long sine wave
time_series_A = temp([1:32]);           % make one test time series from the sine wave
time_series_B = temp([12:43]);          % make another test time series from the sine wave

time_series_A = (time_series_A - mean(time_series_A)) / std(time_series_A);
time_series_B = (time_series_B - mean(time_series_B)) / std(time_series_B);

alphabet_size = 4; % Choose an alphabet size

plot( [time_series_A  time_series_B]) % View the test time series

% Now let us create a SAX representation of the time series
sax_version_of_A = timeseries2symbol(time_series_A,32,8, alphabet_size)
sax_version_of_B = timeseries2symbol(time_series_B,32,8, alphabet_size)

% compute the euclidean distance between the time series
euclidean_distance_A_and_B = sqrt(sum((time_series_A - time_series_B).^2))

% compute the lower bounding distance between the time series
min_dist(sax_version_of_A, sax_version_of_B, alphabet_size,4)
