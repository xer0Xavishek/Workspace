#include <iostream>
#include <string> // Add the <string> header
using namespace std;
class DummyClass{
public:
    void setName(string nme){
    name=nme;
}
string getName(){
return name;
}

private:
    string name;
};


int main(){
DummyClass tarekobject;
tarekobject.setName("Avishek Biswas");
cout<<tarekobject.getName();
return 0;
}
