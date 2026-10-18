Set sh = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
project = fso.GetParentFolderName(WScript.ScriptFullName)
pythonExe = project & "\venv\Scripts\python.exe"
If Not fso.FileExists(pythonExe) Then
  MsgBox "Could not find the project virtual environment here:" & vbCrLf & pythonExe, vbExclamation, "Underwater App"
  WScript.Quit 1
End If
sh.CurrentDirectory = project
sh.Environment("PROCESS")("YOLO_CONFIG_DIR") = project & "\Ultralytics"
sh.Run Chr(34) & pythonExe & Chr(34) & " app.py", 0, False
WScript.Sleep 10000
sh.Run "http://127.0.0.1:5000", 1, False

