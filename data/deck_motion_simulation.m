% =========================================================================
% 航空母舰/大型舰船甲板六自由度运动模型与着舰点动力学仿真
% 模型基础: Fossen 标准波浪成形滤波器与着舰点空间运动学变换
% 海况等级: Sea State 4 (中等海况, H_1/3 = 2.5m, 风速约 20 节)
% =========================================================================
clear; clc; close all;

%% 1. 仿真参数设定
dt = 0.02;                 % 采样周期 (s) -> 50 Hz
t = 0:dt:150;              % 仿真总时长 150 秒
N = length(t);
fs = 1/dt;

s = tf('s');

%% 2. 4级海况 (Sea State 4) 传递函数定义
% 沉浮通道 (Heave: z, 单位: m)
zeta_z  = 0.25;  omega_z  = 0.60;  sigma_z  = 0.95;   % 目标标准差 0.95 m
K_z = 2 * sigma_z * sqrt(zeta_z * omega_z);
G_z = (K_z * s) / (s^2 + 2*zeta_z*omega_z*s + omega_z^2);

% 纵摇通道 (Pitch: theta, 单位: rad)
zeta_th = 0.20;  omega_th = 0.55;  sigma_th = deg2rad(1.25); % 目标标准差 1.25 deg
K_th = 2 * sigma_th * sqrt(zeta_th * omega_th);
G_th = (K_th * s) / (s^2 + 2*zeta_th*omega_th*s + omega_th^2);

% 横摇通道 (Roll: phi, 单位: rad)
zeta_ph = 0.08;  omega_ph = 0.40;  sigma_ph = deg2rad(3.0);  % 目标标准差 3.0 deg
K_ph = 2 * sigma_ph * sqrt(zeta_ph * omega_ph);
G_ph = (K_ph * s) / (s^2 + 2*zeta_ph*omega_ph*s + omega_ph^2);

%% 3. 高斯白噪声激励输入
rng(2026); % 固定随机数种子确保可复现性
w_z  = randn(1, N) / sqrt(dt);
w_th = randn(1, N) / sqrt(dt);
w_ph = randn(1, N) / sqrt(dt);

%% 4. 时域响应求解
z   = lsim(G_z,  w_z,  t)';            % 质心沉浮位移 (m)
th  = lsim(G_th, w_th, t)';            % 纵摇角 (rad)
ph  = lsim(G_ph, w_ph, t)';            % 横摇角 (rad)

%% 5. 理想着舰点 (第3根阻拦索中心) 空间坐标变换
% 空间几何坐标 (相对于船体质心 CG)
xd = -105.0;   % m (位于质心后方 105 米)
yd = 4.5;      % m (偏向右舷/斜角甲板偏移 4.5 米)
zd = 6.2;      % m (甲板高度高于质心 6.2 米)

% 着舰点空间垂向合成位移及分量
Z_heave       = z;
Z_pitch_lever = -xd * th;              % 纵摇杠杆贡献 (-x_d * theta)
Z_roll_lever  = yd * ph;               % 横摇杠杆贡献 (y_d * phi)
Z_deck        = Z_heave + Z_pitch_lever + Z_roll_lever; % 着舰点总垂直位移

% 着舰点水平面扰动
X_deck = -zd * th;                     % 纵向位移扰动 (m)
Y_deck = zd * ph;                      % 横向位移扰动 (m)

% 物理平滑速度计算
tau = 0.05;
H_diff = s / (tau*s + 1);
dZ_deck = lsim(H_diff, Z_deck, t)';

%% 6. 绘图与结果可视化
% 图 1: 各自由度时域响应与着舰点合成
figure('Color', 'w', 'Position', [50, 40, 1150, 850], 'Name', '甲板运动时域响应');

subplot(4, 1, 1);
plot(t, z, 'Color', [0.0 0.45 0.74], 'LineWidth', 1.4);
grid on; ylabel('沉浮 z (m)', 'FontSize', 10, 'FontWeight', 'bold');
title('航母甲板六自由度运动与着舰点动态响应仿真 (Sea State 4)', 'FontSize', 13, 'FontWeight', 'bold');
xlim([0 150]);

subplot(4, 1, 2);
plot(t, rad2deg(th), 'Color', [0.85 0.33 0.10], 'LineWidth', 1.4);
grid on; ylabel('纵摇 \theta (deg)', 'FontSize', 10, 'FontWeight', 'bold');
xlim([0 150]);

subplot(4, 1, 3);
plot(t, rad2deg(ph), 'Color', [0.47 0.67 0.19], 'LineWidth', 1.4);
grid on; ylabel('横摇 \phi (deg)', 'FontSize', 10, 'FontWeight', 'bold');
xlim([0 150]);

subplot(4, 1, 4);
plot(t, Z_deck, 'Color', [0.49 0.18 0.56], 'LineWidth', 1.6, 'DisplayName', '着舰点总垂向位移 Z_{deck}');
hold on;
plot(t, Z_pitch_lever, '--', 'Color', [0.85 0.33 0.10 0.8], 'LineWidth', 1.1, 'DisplayName', '纵摇杠杆贡献 (-x_d\cdot\theta)');
plot(t, Z_heave, ':', 'Color', [0.0 0.45 0.74 0.8], 'LineWidth', 1.1, 'DisplayName', '质心沉浮贡献 (z)');
grid on; xlabel('时间 t (s)', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('着舰点垂向位移 (m)', 'FontSize', 10, 'FontWeight', 'bold');
legend('Location', 'northeast', 'FontSize', 9);
xlim([0 150]);

% 图 2: 频域响应与空间相平面
figure('Color', 'w', 'Position', [70, 50, 1150, 750], 'Name', '频域与空间相平面特性');

subplot(2, 2, 1);
w_vec = logspace(-2, 1, 500);
[mag_z, ~]  = bode(G_z, w_vec);
[mag_th, ~] = bode(G_th, w_vec);
[mag_ph, ~] = bode(G_ph, w_vec);
semilogx(w_vec, 20*log10(squeeze(mag_z)), 'Color', [0.0 0.45 0.74], 'LineWidth', 1.5, 'DisplayName', '沉浮 G_z(s)');
hold on;
semilogx(w_vec, 20*log10(squeeze(mag_th)), 'Color', [0.85 0.33 0.10], 'LineWidth', 1.5, 'DisplayName', '纵摇 G_\theta(s)');
semilogx(w_vec, 20*log10(squeeze(mag_ph)), 'Color', [0.47 0.67 0.19], 'LineWidth', 1.5, 'DisplayName', '横摇 G_\phi(s)');
grid on; xlabel('角频率 \omega (rad/s)', 'FontSize', 10);
ylabel('幅值增益 (dB)', 'FontSize', 10);
title('(a) 传递函数幅频特性 (Bode Magnitude)', 'FontSize', 11, 'FontWeight', 'bold');
legend('Location', 'southwest', 'FontSize', 9);

subplot(2, 2, 2);
L = N;
win = 0.5 * (1 - cos(2*pi*(0:L-1)/(L-1))); % Hanning 窗
Z_win = (Z_deck - mean(Z_deck)) .* win;
Y_fft = fft(Z_win);
P2 = (abs(Y_fft)/L).^2;
P1 = P2(1:floor(L/2)+1);
P1(2:end-1) = 2*P1(2:end-1);
f_axis = fs*(0:(L/2))/L;
plot(f_axis, 10*log10(P1 + eps), 'Color', [0.49 0.18 0.56], 'LineWidth', 1.4);
grid on; xlabel('频率 f (Hz)', 'FontSize', 10);
ylabel('功率谱估计 (dB)', 'FontSize', 10);
title('(b) 着舰点垂向位移功率谱估计 (PSD)', 'FontSize', 11, 'FontWeight', 'bold');
xlim([0 0.35]);

subplot(2, 2, 3);
plot(Z_deck, dZ_deck, 'Color', [0.15 0.25 0.55 0.7], 'LineWidth', 1.1);
hold on;
rectangle('Position', [-3.0, -1.5, 6.0, 3.0], 'EdgeColor', 'r', 'LineStyle', '--', 'LineWidth', 1.6);
plot(0, 0, 'r+', 'MarkerSize', 10, 'LineWidth', 1.5);
grid on; xlabel('垂向位移 Z_{deck} (m)', 'FontSize', 10);
ylabel('垂向速度 dZ_{deck}/dt (m/s)', 'FontSize', 10);
title('(c) 着舰点相平面轨迹与安全窗口 (红框)', 'FontSize', 11, 'FontWeight', 'bold');

subplot(2, 2, 4);
scatter3(X_deck, Y_deck, Z_deck, 18, t, 'filled');
colormap(jet);
cb = colorbar;
cb.Label.String = '时间 t (s)';
grid on; xlabel('纵向扰动 \DeltaX (m)', 'FontSize', 10);
ylabel('横向扰动 \DeltaY (m)', 'FontSize', 10);
zlabel('垂向扰动 \DeltaZ (m)', 'FontSize', 10);
title('(d) 着舰点空间三维位移扰动轨迹', 'FontSize', 11, 'FontWeight', 'bold');
view(45, 25);
