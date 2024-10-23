close all
figure
plot(XY.signals(1).values, XY.signals(2).values, 'LineWidth', 2)
grid on
hold on 
p=plot(XY_nnLin.signals(1).values, XY.signals(2).values, 'LineWidth',2);
set(gca,'FontSize',12)
axis([-0.02, 1.005, -1., 0.02]);
xlabel('$h_{\rm eq}$', 'Interpreter', 'latex', 'FontSize', 20)
ylabel('$\dot{h}_{\rm eq}$', 'Interpreter', 'latex', 'FontSize', 20)
