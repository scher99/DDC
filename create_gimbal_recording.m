
% plot all signals in out
close all
clc

t = out.tout;
pitchrate_cmds = out.cmds_simout.signals.values(:,1);
yawrate_cmds   = out.cmds_simout.signals.values(:,2);

wx = out.meas_out.signals.values(:,1);
wy = out.meas_out.signals.values(:,2);
wz = out.meas_out.signals.values(:,3);
p  = out.meas_out.signals.values(:,4);
q  = out.meas_out.signals.values(:,5);
r  = out.meas_out.signals.values(:,6);
theta = out.meas_out.signals.values(:,7);
psi = out.meas_out.signals.values(:,8);

thetadot = out.relvel_out.signals.values(:,1);
psidot = out.relvel_out.signals.values(:,2);

t_mot_p = out.torques_out.signals.values(:,1);
t_mot_y = out.torques_out.signals.values(:,2);

acc_pitch = out.acc_simout.signals.values(:,1);
acc_yaw   = out.acc_simout.signals.values(:,2);

T = table(t,p,q,r,wy,theta,thetadot,t_mot_p,wz,psi,psidot,t_mot_y,pitchrate_cmds,yawrate_cmds,acc_pitch,acc_yaw);
writetable(T,'tabledata.txt');
disp('done writing file')