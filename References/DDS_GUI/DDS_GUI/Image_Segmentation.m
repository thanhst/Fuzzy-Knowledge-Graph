function varargout = Image_Segmentation(varargin)
% IMAGE_SEGMENTATION MATLAB code for Image_Segmentation.fig
%      IMAGE_SEGMENTATION, by itself, creates a new IMAGE_SEGMENTATION or raises the existing
%      singleton*.
%
%      H = IMAGE_SEGMENTATION returns the handle to a new IMAGE_SEGMENTATION or the handle to
%      the existing singleton*.
%
%      IMAGE_SEGMENTATION('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in IMAGE_SEGMENTATION.M with the given input arguments.
%
%      IMAGE_SEGMENTATION('Property','Value',...) creates a new IMAGE_SEGMENTATION or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before Image_Segmentation_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to Image_Segmentation_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help Image_Segmentation

% Last Modified by GUIDE v2.5 28-Sep-2015 16:21:15

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @Image_Segmentation_OpeningFcn, ...
                   'gui_OutputFcn',  @Image_Segmentation_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before Image_Segmentation is made visible.
function Image_Segmentation_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to Image_Segmentation (see VARARGIN)

% Choose default command line output for Image_Segmentation
handles.output = hObject;
set(handles.org_img_panel, 'visible', 'off');
% Update handles structure
guidata(hObject, handles);


% UIWAIT makes Image_Segmentation wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = Image_Segmentation_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in btn_open.
function btn_open_Callback(hObject, eventdata, handles)
% hObject    handle to btn_open (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
    [filename, pathname] = uigetfile({'*.jpg'}, 'Pick a deltal X - ray image');
 

if filename ~= 0
    set(handles.open_file,'string', filename);
   
    IM = imread([pathname, filename]);
    
%     imfinfo([pathname, filename])
    
    set(handles.org_img_panel, 'visible', 'on');
        
    axes(handles.original_img);
    imshow(IM);
    
    handles.num_of_data_point = size(IM(:, : , 1), 1) * size(IM(:, : , 1), 2);
    handles.filename = filename;
    
%     set(handles.btn_segment, 'enable', 'on');
    
    set(handles.open_file,'UserData',IM);
else
    set(handles.open_file,'string', 'No picture selected');
    
    set(handles.org_img_panel, 'visible', 'off');
end
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function open_file_CreateFcn(hObject, eventdata, handles)
% hObject    handle to open_file (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes on button press in btn_analyze.
function btn_analyze_Callback(hObject, eventdata, handles)
% hObject    handle to btn_analyze (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global h;
select=get(handles.open_file,'UserData');
checkString=get(handles.open_file,'String');
if(strcmp(checkString,'No picture selected'))
    msgbox('No picture selected', 'Error', 'error');   
else    
    h = waitbar(0,'Please wait...');
    %string = strsplit(select);
    %num=str2double(string(1,1));
    [ disease ] = Analyze(select);
    switch(disease)
        case 1
            str=sprintf('CRACKED TEETH\nRecommendation:\nGo to the dentist.');
        case 2
            str=sprintf('HIDDEN TEETH\nRecommendation:\nGo to the dentist.');
        case 3
            str=sprintf('CAVITIES\nRecommendation:\nGo to the dentist..');
        case 4
            str=sprintf('MISSING TEETH\nRecommendation:\nGo to the dentist.');
        case 5
            str=sprintf('PERIODONTISTS\nRecommendation:\nGo to the dentist.');
        otherwise
            str=sprintf('System can not detect disease from this image');          
    end
    set(handles.edit_result,'String', str);
    close(h);
end
guidata(hObject, handles);

% --- Executes on button press in btn_about.
function btn_about_Callback(hObject, eventdata, handles)
% hObject    handle to btn_about (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
str1 = sprintf('DENTAL DIAGNOSIS SYSTEM\n\n');
str2 = sprintf('Dental Diagnosis System helps dentists in decision making by automatic analyzing X-ray and giving result.\n\n');
str3 = sprintf('WARNING: Result retrieved from this system is NOT 100%% guaranteed correctly.\n');
str4 = sprintf('This is just reference for the dentists to make their decision.\n\n');
str5 = sprintf('\t\t\tHanoi - 12/2015.');
msgbox([str1, str2, str3, str4, str5], 'Help', 'help');

% --- Executes during object creation, after setting all properties.
function btn_about_CreateFcn(hObject, eventdata, handles)
% hObject    handle to btn_about (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called



function edit_result_Callback(hObject, eventdata, handles)
% hObject    handle to edit_result (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit_result as text
%        str2double(get(hObject,'String')) returns contents of edit_result as a double


% --- Executes during object creation, after setting all properties.
function edit_result_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit_result (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in btn_save.
function btn_save_Callback(hObject, eventdata, handles)
% hObject    handle to btn_save (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
checkString=get(handles.feature,'String');
checkString2=get(handles.label,'String');
if(isempty(checkString) || isempty(checkString2))
    msgbox('Extracted features or Label can not be blank.', 'Error', 'error'); 
else
    feature=get(handles.feature,'UserData');
    %feature=[42.719 156.4 113.04 0.41599 0.019287];
    data=load('dental.txt');
    record_number=max(data(:,1));
    order=record_number+1;
    update(1,1)=order;
    update(1,2:6)=feature;
    label=get(handles.label,'String');
    update(1,7)=str2num(label);
    dlmwrite('dental.txt',update,'-append');
    msgbox('Database updated.', 'Success', 'help');  
end
guidata(hObject, handles);


function feature_Callback(hObject, eventdata, handles)
% hObject    handle to feature (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of feature as text
%        str2double(get(hObject,'String')) returns contents of feature as a double


% --- Executes during object creation, after setting all properties.
function feature_CreateFcn(hObject, eventdata, handles)
% hObject    handle to feature (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function open_file2_Callback(hObject, eventdata, handles)
% hObject    handle to open_file2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of open_file2 as text
%        str2double(get(hObject,'String')) returns contents of open_file2 as a double


% --- Executes during object creation, after setting all properties.
function open_file2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to open_file2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in bnt_open2.
function bnt_open2_Callback(hObject, eventdata, handles)
% hObject    handle to bnt_open2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
[filename2, pathname2] = uigetfile({'*.jpg'}, 'Pick a deltal X - ray image');
 

if filename2 ~= 0
    set(handles.open_file2,'string', filename2);
   
    IM = imread([pathname2, filename2]);
    
%     imfinfo([pathname, filename])
    
%     set(handles.org_img_panel, 'visible', 'on');
%         
%     axes(handles.original_img);
%     imshow(IM);
    
    handles.num_of_data_point = size(IM(:, : , 1), 1) * size(IM(:, : , 1), 2);
    handles.filename2 = filename2;
    
%     set(handles.btn_segment, 'enable', 'on');
    
    set(handles.open_file2,'UserData',IM);
else
    set(handles.open_file,'string', 'No picture selected');
    
%     set(handles.org_img_panel, 'visible', 'off');
end
guidata(hObject, handles);


% --- Executes on button press in btn_extract.
function btn_extract_Callback(hObject, eventdata, handles)
% hObject    handle to btn_extract (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
checkString=get(handles.open_file2,'String');
if(strcmp(checkString,'No picture selected'))
    msgbox('You have to seclect a picture to extract features.', 'Error', 'error');   
else    
    select=get(handles.open_file2,'UserData');
    h = waitbar(0,'Please wait...');
    %string = strsplit(select);
    %num=str2double(string(1,1));
    [ features ] = Extract(select);
    set(handles.feature,'UserData',features);
    set(handles.feature,'String',mat2str(features,4));
    waitbar(1);
    pause(1);
    close(h);
end
guidata(hObject, handles);



function label_Callback(hObject, eventdata, handles)
% hObject    handle to label (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of label as text
%        str2double(get(hObject,'String')) returns contents of label as a double


% --- Executes during object creation, after setting all properties.
function label_CreateFcn(hObject, eventdata, handles)
% hObject    handle to label (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in btn_help.
function btn_help_Callback(hObject, eventdata, handles)
% hObject    handle to btn_help (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
str1 = sprintf('UPDATE DATABASE\n\n');
str2 = sprintf('Click Open button to select a picture.\n');
str3 = sprintf('Click >> button to extract dental features.\n');
str4 = sprintf('In "Extracted features and Label" panel, first textbox is features extracted, second textbox is label of dental disease.\n\n');
str5 = sprintf('LABEL:\n1. Cracked Teeth\n2. Hidden Teeth\n3. Cavities Teeth\n4. Missing Teeth\n5. Periodontitis\n');
str6 = sprintf('You need to determine disease and fill number of label in the second textbox.');
msgbox([str1, str2, str3, str4, str5, str6], 'Help', 'help');