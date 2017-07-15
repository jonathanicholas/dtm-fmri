load('/mnt/mandarin2/Public_Data/HCP2014/Stats/MDS_taskwaveforms/task_design_matrix_tfMRI_WM_RL_122ss.mat')

Vm = task_waveforms(1).Vm;
zback = Vm(:,1:4);
twoback = Vm(:,5:8);
rest = Vm(:,9);

stim_design = zeros(1,405);
for t = 1:405
    for z = 1:4
        if zback(t,z) == 1
            stim_design(1,t) = 0.5;
        end
        if twoback(t,z) == 1
            stim_design(1,t) = 1;
        end
    end
    if rest(t) == 1
        stim_design(1,t) = 0;
    end
end
stim_design = stim_design'
save('WM_RL_stimdesign.txt','stim_design','-ascii')
