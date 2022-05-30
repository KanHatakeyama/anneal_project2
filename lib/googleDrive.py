from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import joblib
import time

#mount
gauth = GoogleAuth()
gauth.LocalWebserverAuth()
drive = GoogleDrive(gauth)


def drive_save(filename,data=None):
    #save
    
    #delete old one
    file_list = drive.ListFile().GetList()
    for f in file_list:
        if f["title"]==filename:
            f.Delete()

    #save
    f = drive.CreateFile({'title': filename})
    
    if data is not None:
        f.SetContentString(data)
    else:
        f.SetContentFile(filename)
        
    f.Upload()

#load
def drive_load(filename,string=True):
    file_list = drive.ListFile().GetList()
    for f in file_list:
        if f["title"]==filename:
            break

    for i in range(100):
        try:
            if string:
                data = f.GetContentString()
                break
            else:
                data=f.GetContentFile("got_"+filename)

        except:
            time.sleep(5)
        
    return data      

def auto_anneal(qubo):
    return submit_anneal_job(qubo)

def submit_anneal_job(qubo):
    q_path="q.bin"
    joblib.dump(-qubo,q_path,compress=9)

    #submit job
    drive_save(q_path)
    drive_save("status","submit")

    #wait for anneal
    for i in range(10**3):
    #while True:
        #print("wait...")
        status=drive_load("status",string=True)
        print(i,status)
        if status=="done":
            print("done!")
            break
        time.sleep(10)

        if i==100:
            raise ValueError("server error!")
            
    #load fp
    drive_load("fp.bin",string=False)
    found_fp=joblib.load("got_fp.bin")
    
    return found_fp
