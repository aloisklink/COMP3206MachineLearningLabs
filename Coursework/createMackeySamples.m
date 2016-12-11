function [ X T ] = createMackeySamples( numberOfSamples )
%createMackeySamples Creates `numberOfSamples` Mackey-Glass samples

% The following code requires files from:
% <http://uk.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/24390/versions/1/download/zip>
path(path,'../libraries/mackeyglass');
%% Input parameters
a        = 0.2;     % value for a in eq (1)
b        = 0.1;     % value for b in eq (1)
tau      = 17;		% delay constant in eq (1)
x0       = 1.2;		% initial condition: x(t=0)=x0
deltat   = 0.1;	    % time step size (which coincides with the integration step)
sample_n = 2000;	% total no. of samples, excluding the given initial condition
interval = 1;	    % output is printed at every 'interval' time steps


time = 0;
index = 1;
history_length = floor(tau/deltat);
x_history = zeros(history_length, 1); % here we assume x(t)=0 for -tau <= t < 0
x_t = x0;

X = zeros(sample_n+1, 1); % vector of all generated x samples
T = zeros(sample_n+1, 1); % vector of time samples

for i = 1:sample_n+1,
    X(i) = x_t;
    if tau == 0,
        x_t_minus_tau = 0.0;
    else
        x_t_minus_tau = x_history(index);
    end

    x_t_plus_deltat = mackeyglass_rk4(x_t, x_t_minus_tau, deltat, a, b);

    if (tau ~= 0),
        x_history(index) = x_t_plus_deltat;
        index = mod(index, history_length)+1;
    end
    time = time + deltat;
    T(i) = time;
    x_t = x_t_plus_deltat;
end

data.values = X;
data.time = T;

end

