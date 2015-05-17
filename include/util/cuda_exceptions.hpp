#pragma once

#include <exception>
#include <string>

namespace cuda {

class cuda_exception : std::exception
{
public:
    explicit cuda_exception(const std::string& what_arg)
            : why{what_arg} {};

    explicit cuda_exception(const char* what_arg)
            : why{std::string(what_arg)} {};

    virtual ~cuda_exception() throw() {}

    virtual const char* what() const throw() {
        return why.c_str();
    }

private:
    std::string why;
};


}