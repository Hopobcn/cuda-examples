#pragma once

#include <iterator>
#include <type_traits>
#include "device_callable.hpp"

namespace  cuda { namespace util { namespace lang {

namespace detail {

template <typename T>
struct range_iter_base : std::iterator<std::input_iterator_tag, T> {
    HOST_DEVICE_CALLABLE_INLINE
    range_iter_base(T current) : current(current) { }

    HOST_DEVICE_CALLABLE_INLINE
    T operator *() const { return current; }

    HOST_DEVICE_CALLABLE_INLINE
    T const* operator ->() const { return &current; }

    HOST_DEVICE_CALLABLE_INLINE
    range_iter_base& operator ++() {
        ++current;
        return *this;
    }

    HOST_DEVICE_CALLABLE_INLINE
    range_iter_base operator ++(int) {
        auto copy = *this;
        ++*this;
        return copy;
    }

    HOST_DEVICE_CALLABLE_INLINE
    bool operator ==(range_iter_base const& other) const {
        return current == other.current;
    }

    HOST_DEVICE_CALLABLE_INLINE
    bool operator !=(range_iter_base const& other) const {
        return not (*this == other);
    }

protected:
    T current;
};

} // namespace detail

template <typename T>
struct range_proxy {
    struct iter : detail::range_iter_base<T> {
        HOST_DEVICE_CALLABLE_INLINE
        iter(T current) : detail::range_iter_base<T>(current) { }
    };

    struct step_range_proxy {
        struct iter : detail::range_iter_base<T> {
            HOST_DEVICE_CALLABLE_INLINE
            iter(T current, T step)
                    : detail::range_iter_base<T>(current), step(step) { }

            using detail::range_iter_base<T>::current;

            HOST_DEVICE_CALLABLE_INLINE
            iter& operator ++() {
                current += step;
                return *this;
            }

            HOST_DEVICE_CALLABLE_INLINE
            iter operator ++(int) {
                auto copy = *this;
                ++*this;
                return copy;
            }

            // Loses commutativity. Iterator-based ranges are simply broken. :-(
            HOST_DEVICE_CALLABLE_INLINE
            bool operator ==(iter const& other) const {
                return step > 0 ? current >= other.current
                                : current < other.current;
            }

            HOST_DEVICE_CALLABLE_INLINE
            bool operator !=(iter const& other) const {
                return not (*this == other);
            }

        private:
            T step;
        };

        HOST_DEVICE_CALLABLE_INLINE
        step_range_proxy(T begin, T end, T step)
                : begin_(begin, step), end_(end, step) { }

        HOST_DEVICE_CALLABLE_INLINE
        iter begin() const { return begin_; }

        HOST_DEVICE_CALLABLE_INLINE
        iter end() const { return end_; }

    private:
        iter begin_;
        iter end_;
    };

    HOST_DEVICE_CALLABLE_INLINE
    range_proxy(T begin, T end) : begin_(begin), end_(end) { }

    HOST_DEVICE_CALLABLE_INLINE
    step_range_proxy step(T step) {
        return {*begin_, *end_, step};
    }

    HOST_DEVICE_CALLABLE_INLINE
    iter begin() const { return begin_; }

    HOST_DEVICE_CALLABLE_INLINE
    iter end() const { return end_; }

private:
    iter begin_;
    iter end_;
};

template <typename T>
struct infinite_range_proxy {
    struct iter : detail::range_iter_base<T> {
        HOST_DEVICE_CALLABLE_INLINE
        iter(T current = T()) : detail::range_iter_base<T>(current) { }

        HOST_DEVICE_CALLABLE_INLINE
        bool operator ==(iter const&) const { return false; }

        HOST_DEVICE_CALLABLE_INLINE
        bool operator !=(iter const&) const { return true; }
    };

    struct step_range_proxy {
        struct iter : detail::range_iter_base<T> {
            HOST_DEVICE_CALLABLE_INLINE
            iter(T current = T(), T step = T())
                    : detail::range_iter_base<T>(current), step(step) { }

            using detail::range_iter_base<T>::current;

            HOST_DEVICE_CALLABLE_INLINE
            iter& operator ++() {
                current += step;
                return *this;
            }

            HOST_DEVICE_CALLABLE_INLINE
            iter operator ++(int) {
                auto copy = *this;
                ++*this;
                return copy;
            }

            HOST_DEVICE_CALLABLE_INLINE
            bool operator ==(iter const&) const { return false; }

            HOST_DEVICE_CALLABLE_INLINE
            bool operator !=(iter const&) const { return true; }

        private:
            T step;
        };

        HOST_DEVICE_CALLABLE_INLINE
        step_range_proxy(T begin, T step) : begin_(begin, step) { }

        HOST_DEVICE_CALLABLE_INLINE
        iter begin() const { return begin_; }

        HOST_DEVICE_CALLABLE_INLINE
        iter end() const { return  iter(); }

    private:
        iter begin_;
    };

    HOST_DEVICE_CALLABLE_INLINE
    infinite_range_proxy(T begin) : begin_(begin) { }

    HOST_DEVICE_CALLABLE_INLINE
    step_range_proxy step(T step) {
        return step_range_proxy(*begin_, step);
    }

    HOST_DEVICE_CALLABLE_INLINE
    iter begin() const { return begin_; }

    HOST_DEVICE_CALLABLE_INLINE
    iter end() const { return iter(); }

private:
    iter begin_;
};

template <typename T>
HOST_DEVICE_CALLABLE_INLINE
range_proxy<T> range(T begin, T end) {
    return {begin, end};
}

template <typename T>
HOST_DEVICE_CALLABLE_INLINE
infinite_range_proxy<T> range(T begin) {
    return {begin};
}

namespace traits {

template <typename C>
struct has_size {
    template <typename T>
    static constexpr auto check(T*) ->
            typename std::is_integral<
                    decltype(std::declval<T const>().size())>::type;

    template <typename>
    static constexpr auto check(...) -> std::false_type;

    using type = decltype(check<C>(0));
    static constexpr bool value = type::value;
};

} // namespace traits

template <typename C, typename = typename std::enable_if<traits::has_size<C>::value>>
HOST_DEVICE_CALLABLE_INLINE
auto indices(C const& cont) -> range_proxy<decltype(cont.size())> {
    return {0, cont.size()};
}

template <typename T, std::size_t N>
HOST_DEVICE_CALLABLE_INLINE
range_proxy<std::size_t> indices(T (&)[N]) {
    return {0, N};
}

template <typename T>
range_proxy<typename std::initializer_list<T>::size_type>
HOST_DEVICE_CALLABLE_INLINE
indices(std::initializer_list<T>&& cont) {
    return {0, cont.size()};
}

}}} // namespace cuda::util::lang