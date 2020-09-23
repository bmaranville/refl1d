Set oWS = WScript.CreateObject("WScript.Shell") > CreateShortcut.vbs
dim scriptdir 
scriptdir = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName) 
sLinkFile = "%userprofile%\Desktop\Refl1D.lnk" 
Set oLink = oWS.CreateShortcut(sLinkFile) 
oLink.TargetPath = scriptdir &"\pythonw.exe" 
oLink.Arguments = " -m refl1d.main --edit" 
oLink.WorkingDirectory = scriptdir 
oLink.Description = "Refl1D" 
oLink.IconLocation = scriptdir &"\refl1d.ico" 
oLink.Save 