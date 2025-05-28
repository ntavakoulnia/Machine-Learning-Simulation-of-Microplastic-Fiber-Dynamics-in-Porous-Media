close all
clear
clc
% Use the current directory
rootDirectory = 'D:\Simulations\GOODRESULTS1000------NONUNIFORM\1000Re=10GOODSIM';
% Collect folders in '0.*' format
folders = dir(fullfile(rootDirectory, '0.1*'));
% % Initialize lists to store data for training
nfilament = length(folders);
npoints = 50;
ntimestep = 250;
X_train = -ones(ntimestep*nfilament,2);
y_train = -ones(ntimestep*nfilament,npoints*2);
% Iterate through folders
for i = 1:nfilament
    rownum = ntimestep*(i-1)+1;
    foldername = folders(i).name;
    % Read VTK files in viz_IB2d folder
    vtkFilePath = fullfile(rootDirectory, foldername, 'viz_IB2d', 'lagPtsConnect.*.vtk');
    vtkFiles = dir(vtkFilePath);
    % Exclude folder '0.164' and collect data from all other folders
   % if ~strcmp(foldername, '0.1548')
        n_itr = ntimestep;
        if(length(vtkFiles) < ntimestep)
            n_itr = length(vtkFiles);
        end
        for j = 1:n_itr
            vtkFileName = fullfile(rootDirectory, foldername, 'viz_IB2d', vtkFiles(j).name);
            % Read the VTK file
            vtkData = vtkRead(vtkFileName);
            % Extract x, y, and time data from VTK file
            y_train(rownum,1:npoints) = vtkData.points(1:50, 1);
            y_train(rownum,npoints+1:end) = vtkData.points(1:50, 2)';
            X_train(rownum,:) = [str2double(foldername), vtkData.fieldData.TIME];
            rownum = rownum + 1;
        end
    end
%end
X_train_Table = array2table(X_train, 'VariableNames', {'Y0', 'TIME'});
writetable(X_train_Table, 'x_train.csv');
columnLabels = cell(1, 2*npoints);
for i = 1:npoints
    columnLabels{i} = sprintf('X%d', i);
    columnLabels{i+50} = sprintf('Y%d', i);
end
y_train_Table = array2table(y_train, 'VariableNames', columnLabels);
writetable(y_train_Table, 'y_train.csv');