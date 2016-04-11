#include <string>
#include <vector>
#include<glob.h>

using namespace std;

vector<string> glob(const string &pattern)
{
        glob_t glob_result;
        glob(pattern.c_str(), GLOB_TILDE, NULL, &glob_result);
        vector<string>filename;
        if (glob_result.gl_pathc > 0)
        {
                filename.resize(glob_result.gl_pathc);
                for (int i=0; i<glob_result.gl_pathc; i++)
                        filename[i] = string(glob_result.gl_pathv[i]);
        }
        else
                filename.resize(0);
        globfree(&glob_result);
        return filename;
}

