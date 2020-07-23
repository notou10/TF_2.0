import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding = 'utf-8')

def initial(name, age, gender = True):
    print("%s아고 합니다" % name)
    print("%d 살 입니다" % age)
    if gender:
        print("남자입니다")
    else:
        print("여자입니다")

initial("동균",23,1)
initial("기창",22)
initial("수빈",20,0)