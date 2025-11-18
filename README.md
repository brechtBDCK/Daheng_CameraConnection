The Core of the Problem
Different Operating Systems, Different Libraries:

Windows uses dynamic-link libraries with a .dll extension. The Windows SDK you installed provides these.

Linux (including WSL) uses shared object libraries with a .so extension.

The error message Cannot find libgxiapi.so is your Linux environment telling you it cannot find the required Linux library.

WSL is a Separate Linux Machine:

Think of WSL as a virtual machine running a full Linux kernel. The programs you run inside it (like your Python interpreter) are Linux programs. They cannot use the Windows .dll files from your host system.

Even though WSL2 has excellent hardware integration (it can access USB ports and network interfaces from the host), it needs its own Linux drivers and libraries to communicate with that hardware.

The secondary NameError: name 'dll' is not defined is just a consequence of the first error. The gxipy code failed to load the .so file, so the variable it was supposed to be assigned to (dll) was never created, causing a crash later on.

The Solution: Install the Linux SDK inside WSL
You need to download and install the Linux version of the Daheng Galaxy SDK directly within your WSL distribution.

Here is a step-by-step guide:

Step 1: Download the Linux SDK

Go to the Daheng Imaging Download Center.

Find the SDK for Linux. The file will likely have a name like Galaxy_Linux-x86_Gige-U3_... and will probably be a .run file or a .tar.gz archive.

Download it to your Windows Downloads folder.

Step 2: Move the Installer into WSL

Open your WSL terminal. Your Windows C: drive is automatically mounted under /mnt/c/. You can copy the file from your Windows Downloads folder into your WSL home directory.

Bash

# Example: Replace 'YourWindowsUsername' with your actual username
cp /mnt/c/Users/YourWindowsUsername/Downloads/Galaxy_Linux*.run ~/
Navigate to your home directory:

Bash

cd ~
Step 3: Make the Installer Executable

You need to give the .run file permission to be executed.

Bash

# Replace the filename with the one you downloaded
chmod +x Galaxy_Linux-x86_Gige-U3_*.run
Step 4: Run the Installer

Run the installer using sudo because it needs to install files into system directories.

Bash

# Replace the filename with the one you downloaded
sudo ./Galaxy_Linux-x86_Gige-U3_*.run
Follow the on-screen instructions. Install it to the default location, which is usually somewhere like /opt/DxImageCard/Galaxy_SDK/.

Step 5: Set the Library Path Environment Variable (Very Important!)

This is the step that directly solves the Cannot find libgxiapi.so error. The Linux system needs to know where to look for the .so files you just installed. You do this by setting the LD_LIBRARY_PATH environment variable.

The Daheng libraries are typically installed in /opt/DxImageCard/Galaxy_SDK/lib/.

Set it for your current session (for testing):

Bash

export LD_LIBRARY_PATH=/opt/DxImageCard/Galaxy_SDK/lib:$LD_LIBRARY_PATH
Make it permanent: To avoid typing the command above every time you open a new terminal, add it to your shell's startup file (e.g., ~/.bashrc).

Bash

echo 'export LD_LIBRARY_PATH=/opt/DxImageCard/Galaxy_SDK/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
This appends the command to your .bashrc file and then re-loads the file to apply the change to your current session.




## Next pass through the usb camera to wsl (powershell with admin rights)
usbipd list
usbipd bind --busid 2-18 --force
usbipd attach --wsl --busid 2-18
--> in wsl check with lsusb if the daheng camera is visible
