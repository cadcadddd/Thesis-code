import PySimpleGUI as sg
import numpy as np
import pandas as pd
from pickle import load
from PIL import Image
import os
import traceback
import io

current_dir = os.path.dirname(os.path.abspath(__file__))

# import the dataset
try:
    dd1 = pd.read_excel('testPL.xlsx', sheet_name='testPL')
    df = dd1.copy(deep=True)
except FileNotFoundError:
    sg.popup_error("The data.xlsx file was not found! Please ensure that the file is in the same directory.")
    exit()
except Exception as e:
    sg.popup_error(f"Error reading Excel file:{str(e)}")
    exit()

t = 32
td2 = 8

# define the range of values for each parameter
param_ranges = {
    'slenderness': (0, 80),              #  lf/df
    'fiber_length': (0, 60),             #  lf (mm)
    'volume_fraction': (0, 2),           #  Vf (%)
    'compressive_strength': (22, 66.8),  #  fc' (MPa)
    'rebar_diameter': (8, 20),           #  d (mm)
    'cover_ratio': (0.94, 6.58),         #  c/d
    'bond_length_ratio': (3, 9.4)        #  l/d
}

# Test method correction factor
test_method_factors = {
    'PL: Pullout test': 1.0,
    'BSF: Beam tests for bond in flexural-shear region': 1.27,
    'BPB: Beam tests for bond in pure-bending region': 0.68
}

sg.theme('LightGrey1')

layout = [
    [sg.Text('Developed by Xuhui Zhang, Junzhuo Chen, Ping Yuan, Lei Wang.')],
    [sg.Text('College of Civil Engineering, Xiangtan University')],
    [sg.HorizontalSeparator()],  # Add horizontal separator line
    [
        sg.Column(layout=[
            [sg.Frame(layout=[
                [sg.Text('Slenderness of steel fiber (lf/df)', size=(t, 1)), sg.InputText(key='-SLENDERNESS-', size=(td2, 1)),
                sg.Text('--')],

                [sg.Text('Fiber length (lf) ', size=(t, 1)),
                 sg.InputText( key='-FIBER_LENGTH-',
                                size=(td2, 1)), sg.Text('mm')],

                [sg.Text('Steel fiber volume fraction (Vf)', size=(t, 1)), sg.InputText(key='-VOLUME_FRACTION-', size=(td2, 1)), sg.Text('%')],

                [sg.Text('Concrete compressive strength (fc\')', size=(t, 1)), sg.InputText(key='-COMPRESSIVE_STRENGTH-', size=(td2, 1)), sg.Text('MPa')],

                [sg.Text('Rebar diameter (d)', size=(t, 1)), sg.InputText(key='-REBAR_DIAMETER-', size=(td2, 1)),
                 sg.Text('mm')],

                [sg.Text('Ratio of cover to rebar diameter(c/d)', size=(t, 1)), sg.InputText(key='-COVER_RATIO-', size=(td2, 1)),
                 sg.Text('--')],

                [sg.Text('Ratio of bond length to rebar diameter(l/d)', size=(t, 1)), sg.InputText( key='-BOND_LENGTH_RATIO-',
                                                                                  size=(td2, 1)),
                 sg.Text('--')],

                [sg.Text('Test method', size=(t, 1)),
                 sg.Combo(
                     ['PL: Pullout test', 'BSF: Beam tests for bond in flexural-shear region', 'BPB: Beam tests for bond in pure-bending region'],  # 选项列表作为单独参数
                     default_value='PL: Pullout test',
                     key='-TEST_METHOD-',
                     size=(td2, 1)),
                 sg.Text('--')]],

                title='Input parameters',
                relief=sg.RELIEF_SUNKEN,
                expand_x=True, expand_y=True
            )],
        ], justification='left', expand_x=True, expand_y=True),

        sg.Column(layout=[
            [sg.Frame(layout=[
                [sg.Text('0 ≤ lf/df ≤ 80', expand_x=True)],
                [sg.Text('0 mm ≤ lf ≤ 60 mm', expand_x=True)],
                [sg.Text('0 % ≤ Vf ≤ 2 %', expand_x=True)],
                [sg.Text('22 MPa ≤ fc\' ≤ 66.8 MPa', expand_x=True)],
                [sg.Text('8 mm ≤ d ≤ 20 mm', expand_x=True)],
                [sg.Text('0.94 ≤ c/d ≤ 6.58', expand_x=True)],
                [sg.Text('3 ≤ l/d ≤ 9.40', expand_x=True)],
                [sg.Text('', expand_x=True)],
                [sg.Text('Test method options:', font=('Arial', 10, 'bold'), expand_x=True)],
                [sg.Text('PL: Pullout test (factor=1.0)')],
                [sg.Text('BSF: Beam tests for bond in flexural-shear region (factor=1.27)', expand_x=True)],
                [sg.Text('BPB: Beam tests for bond in pure-bending region (factor=0.68)', expand_x=True)]],

                title='Range of application of the models', expand_x=True, expand_y=True)],

        ], justification='center')
    ],
    [sg.Frame(layout=[
        [sg.Text('Bond strength', size=(32, 1)), sg.InputText(key='-OP-', size=(td2, 1), expand_x=True), sg.Text('MPa')]],
        title='Output')],
    [sg.Button('Predict', size=(10, 1)), sg.Button('Cancel', size=(10, 1))],
    [sg.Sizegrip()]
]


# Open the images
try:
    img1_path = os.path.join(current_dir, 'image1.png')
    img2_path = os.path.join(current_dir, 'image2.png')
    print(f"Image 1 path: {img1_path}")
    print(f"Image 2 path: {img2_path}")

    if os.path.exists(img1_path) and os.path.exists(img2_path):
        print("Image file exists")

        # Open image
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)

        print(f"Image 1 Original Size: {img1.size}")
        print(f"Image 2 Original Size: {img2.size}")

        # Adjust image size to fit window
        max_width = 400
        scale_factor1 = max_width / img1.width
        scale_factor2 = max_width / img2.width
        scale_factor = min(scale_factor1, scale_factor2)

        new_size1 = (int(img1.width * scale_factor), int(img1.height * scale_factor))
        new_size2 = (int(img2.width * scale_factor), int(img2.height * scale_factor))

        print(f"Image 1 New dimensions: {new_size1}")
        print(f"Image 2 New dimensions: {new_size2}")

        img1 = img1.resize(new_size1)
        img2 = img2.resize(new_size2)

        # Convert the image to byte data in PNG format
        img1_bytes = io.BytesIO()
        img2_bytes = io.BytesIO()
        img1.save(img1_bytes, format='PNG')
        img2.save(img2_bytes, format='PNG')

        # Add images to the layout
        layout += [
            [sg.Text('Model Feature Importance and Analysis', font=('Arial', 11, 'bold'), justification='center', pad=(0, (15, 5)))],
            [sg.Column([
                [sg.Image(data=img1_bytes.getvalue(), key='-IMG1-')],
                [sg.Text('Parameter global importance', font=('Arial', 9), justification='center')]
            ], element_justification='center'),
                sg.Column([
                    [sg.Image(data=img2_bytes.getvalue(), key='-IMG2-')],
                    [sg.Text('Prediction result analysis', font=('Arial', 9), justification='center')]
                ], element_justification='center')]
        ]

        print("The image has been successfully added to the layout.")
    else:
        error_msg = []
        if not os.path.exists(img1_path):
            error_msg.append(f"image1.png (path: {img1_path})")
        if not os.path.exists(img2_path):
            error_msg.append(f"image2.png (path: {img2_path})")
        error_msg = "The following image files were not found: \n" + "\n".join(error_msg)
        print(error_msg)
        layout += [
            [sg.Text('Image preview not available', font=('Arial', 10), text_color='orange', justification='center', pad=(0, 10))],
            [sg.Text(error_msg, font=('Arial', 8), text_color='gray', justification='center')]
        ]

except Exception as e:
    print(f"Image processing error: {e}")
    traceback.print_exc()

    layout += [
        [sg.Text('Image preview not available', font=('Arial', 10), text_color='orange', justification='center', pad=(0, 10))],
        [sg.Text(f'reason: {str(e)}', font=('Arial', 8), text_color='gray', justification='center')]
    ]

# Load model
try:
    model_path = os.path.join(current_dir, 'main1_model.pkl')
    model = load(open(model_path, 'rb'))
    print("Model loaded successfully!")
except FileNotFoundError:
    sg.popup_error("The model file main1_model.pkl was not found! Please ensure that the file is in the same directory.")
    exit()
except Exception as e:
    sg.popup_error(f"Error loading model: {e}")
    exit()

# Create the Window
window = sg.Window('Explainable machine learning-based prediction of bond strength between steel rebar and steel fiber reinforced concrete', layout,
                   finalize=True,
                   element_justification='center',
                   resizable=True
                   )

# Set window icon
try:
    # You can replace it with your own icon file.
    # window.set_icon(os.path.join(current_dir, 'icon.ico'))
    pass
except:
    print("Unable to set window icon")


# reset function
def reset_inputs():
    window['-SLENDERNESS-'].update('')
    window['-FIBER_LENGTH-'].update('')
    window['-VOLUME_FRACTION-'].update('')
    window['-COMPRESSIVE_STRENGTH-'].update('')
    window['-REBAR_DIAMETER-'].update('')
    window['-COVER_RATIO-'].update('')
    window['-BOND_LENGTH_RATIO-'].update('')
    window['-OP-'].update('')
    window['-TEST_METHOD-'].update('PL: Pullout test')

# event loop
while True:
    event, values = window.read()

    # Close window
    if event in (None, 'Cancel'):
        break

    # reset button
    elif event == '-RESET-':
        reset_inputs()

    # Predict button
    elif event == 'Predict':
        # Verify that all fields have been filled in.
        input_keys = [
            '-SLENDERNESS-',
            '-FIBER_LENGTH-',
            '-VOLUME_FRACTION-',
            '-COMPRESSIVE_STRENGTH-',
            '-REBAR_DIAMETER-',
            '-COVER_RATIO-',
            '-BOND_LENGTH_RATIO-',
            '-TEST_METHOD-'
        ]

        missing_fields = [key for key in input_keys if not values[key]]

        if missing_fields:
            sg.popup("Please fill in all required fields!", title="Incomplete input")
            continue

        try:
            inputs = {
                'slenderness': float(values['-SLENDERNESS-']),
                'fiber_length': float(values['-FIBER_LENGTH-']),
                'volume_fraction': float(values['-VOLUME_FRACTION-']),
                'compressive_strength': float(values['-COMPRESSIVE_STRENGTH-']),
                'rebar_diameter': float(values['-REBAR_DIAMETER-']),
                'cover_ratio': float(values['-COVER_RATIO-']),
                'bond_length_ratio': float(values['-BOND_LENGTH_RATIO-'])
            }

            # Check whether the input value is within the defined range.
            errors = []
            for param, value in inputs.items():
                min_val, max_val = param_ranges[param]
                if value < min_val or value > max_val:
                    param_name = {
                        'slenderness': 'Steel fibre slenderness ratio (lf/df)',
                        'fiber_length': 'Steel fibre length (lf)',
                        'volume_fraction': 'Steel fibre volume fraction (Vf)',
                        'compressive_strength': 'Compressive_strength (fc\')',
                        'rebar_diameter': 'rebar_diameter (d)',
                        'cover_ratio': 'cover_ratio (c/d)',
                        'bond_length_ratio': 'bond_length_ratio (l/d)'
                    }[param]

                    errors.append(f"{param_name}: {value:.2f} (range: {min_val}-{max_val})")

            if errors:
                error_msg = "The following parameters are outside the valid range:\n\n" + "\n".join(errors)
                sg.popup(error_msg, title="Input out of range")
                continue

            input_data = np.array([
                inputs['slenderness'],
                inputs['fiber_length'],
                inputs['volume_fraction'],
                inputs['compressive_strength'],
                inputs['rebar_diameter'],
                inputs['cover_ratio'],
                inputs['bond_length_ratio']
            ]).reshape(1, -1)

            # make predictions
            prediction = model.predict(input_data)[0]

            test_method = values['-TEST_METHOD-']
            factor = test_method_factors.get(test_method, 1.0)
            adjusted_prediction = prediction * factor

            # Display predicted results (rounded to two decimal places)
            window['-OP-'].update(f"{adjusted_prediction:.2f}")

            sg.popup_notify("Predicting success!\nBond strength has been calculated.", title="Complete!", fade_in_duration=500, display_duration_in_ms=3000)

        except ValueError:
            sg.popup("Please enter a valid value!", title="Input Error!")
        except Exception as e:
            print(f"Prediction Error: {e}")
            sg.popup(f"Errors occurring during the prediction process:\n\n{str(e)}", title="Error")

# Clean up temporary image files
try:
    if os.path.exists('image11.png'):
        os.remove('image11.png')
    if os.path.exists('image22.png'):
        os.remove('image22.png')
except:
    pass

# Close window
window.close()