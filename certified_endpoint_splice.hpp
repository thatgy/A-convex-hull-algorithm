#pragma once
#include "hull_scalable.hpp"
#include <stdexcept>
#include <utility>

// Specialized O(1) pointer bridge for two CCW circular hull lists.
// The four tangent vertices MUST ALREADY BE KNOWN. This does not find
// tangents, certify convexity, check separation, or discard interior arcs.
// Callers must certify that the resulting boundary follows these arcs:
// left_upper -> (remaining outer-left arc) -> left_lower -> right_lower
// -> (remaining outer-right arc) -> right_upper -> left_upper.
// It is NOT used as an automatic replacement for the safe heap-chain merge.
namespace scalable {
template<class Real>class CertifiedEndpointSplice {
public:
    using List=CircularList<Point<Real>>;
    using Node=typename List::Node;
private:
    List left_,right_;
    Node*head_{};
public:
    CertifiedEndpointSplice(List&& left,List&& right,
                            Node* left_upper,Node*left_lower,
                            Node*right_upper,Node*right_lower)
      : left_(std::move(left)),right_(std::move(right)),head_(left_upper) {
        if(!left_upper||!left_lower||!right_upper||!right_lower)
            throw std::invalid_argument("all four certified bridge endpoints are required");
        left_lower->next=right_lower;right_lower->prev=left_lower;
        right_upper->next=left_upper;left_upper->prev=right_upper;
    }
    CertifiedEndpointSplice(const CertifiedEndpointSplice&)=delete;
    CertifiedEndpointSplice&operator=(const CertifiedEndpointSplice&)=delete;
    Node*head()const noexcept{return head_;}
};
}
