import torch 


def normalize_bbox(bboxes, pc_range):

    cx = bboxes[..., 0:1]
    cy = bboxes[..., 1:2]
    cz = bboxes[..., 2:3]
    w = bboxes[..., 3:4].log()
    l = bboxes[..., 4:5].log()
    h = bboxes[..., 5:6].log()

    rot = bboxes[..., 6:7]
    if bboxes.size(-1) > 7:
        vx = bboxes[..., 7:8] 
        vy = bboxes[..., 8:9]
        normalized_bboxes = torch.cat(
            (cx, cy, w, l, cz, h, rot.sin(), rot.cos(), vx, vy), dim=-1
        )
    else:
        normalized_bboxes = torch.cat(
            (cx, cy, w, l, cz, h, rot.sin(), rot.cos()), dim=-1
        )
    return normalized_bboxes

def denormalize_bbox(normalized_bboxes, pc_range):
    # rotation 
    rot_sine = normalized_bboxes[..., 6:7]

    rot_cosine = normalized_bboxes[..., 7:8]
    rot = torch.atan2(rot_sine, rot_cosine)

    # center in the bev
    cx = normalized_bboxes[..., 0:1]
    cy = normalized_bboxes[..., 1:2]
    cz = normalized_bboxes[..., 4:5]
   
    # size
    w = normalized_bboxes[..., 2:3]
    l = normalized_bboxes[..., 3:4]
    h = normalized_bboxes[..., 5:6]

    w = w.exp() 
    l = l.exp() 
    h = h.exp() 
    if normalized_bboxes.size(-1) > 8:
         # velocity 
        vx = normalized_bboxes[:, 8:9]
        vy = normalized_bboxes[:, 9:10]
        denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot, vx, vy], dim=-1)
    else:
        denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot], dim=-1)
    return denormalized_bboxes


# Copyright (c) OpenMMLab. All rights reserved.
"""This module defines the :class:`NiceRepr` mixin class, which defines a
``__repr__`` and ``__str__`` method that only depend on a custom ``__nice__``
method, which you must define. This means you only have to overload one
function instead of two.  Furthermore, if the object defines a ``__len__``
method, then the ``__nice__`` method defaults to something sensible, otherwise
it is treated as abstract and raises ``NotImplementedError``.

To use simply have your object inherit from :class:`NiceRepr`
(multi-inheritance should be ok).

This code was copied from the ubelt library: https://github.com/Erotemic/ubelt

Example:
    >>> # Objects that define __nice__ have a default __str__ and __repr__
    >>> class Student(NiceRepr):
    ...    def __init__(self, name):
    ...        self.name = name
    ...    def __nice__(self):
    ...        return self.name
    >>> s1 = Student('Alice')
    >>> s2 = Student('Bob')
    >>> print(f's1 = {s1}')
    >>> print(f's2 = {s2}')
    s1 = <Student(Alice)>
    s2 = <Student(Bob)>

Example:
    >>> # Objects that define __len__ have a default __nice__
    >>> class Group(NiceRepr):
    ...    def __init__(self, data):
    ...        self.data = data
    ...    def __len__(self):
    ...        return len(self.data)
    >>> g = Group([1, 2, 3])
    >>> print(f'g = {g}')
    g = <Group(3)>
"""
import warnings


class NiceRepr:
    """Inherit from this class and define ``__nice__`` to "nicely" print your
    objects.

    Defines ``__str__`` and ``__repr__`` in terms of ``__nice__`` function
    Classes that inherit from :class:`NiceRepr` should redefine ``__nice__``.
    If the inheriting class has a ``__len__``, method then the default
    ``__nice__`` method will return its length.

    Example:
        >>> class Foo(NiceRepr):
        ...    def __nice__(self):
        ...        return 'info'
        >>> foo = Foo()
        >>> assert str(foo) == '<Foo(info)>'
        >>> assert repr(foo).startswith('<Foo(info) at ')

    Example:
        >>> class Bar(NiceRepr):
        ...    pass
        >>> bar = Bar()
        >>> import pytest
        >>> with pytest.warns(None) as record:
        >>>     assert 'object at' in str(bar)
        >>>     assert 'object at' in repr(bar)

    Example:
        >>> class Baz(NiceRepr):
        ...    def __len__(self):
        ...        return 5
        >>> baz = Baz()
        >>> assert str(baz) == '<Baz(5)>'
    """

    def __nice__(self):
        """str: a "nice" summary string describing this module"""
        if hasattr(self, '__len__'):
            # It is a common pattern for objects to use __len__ in __nice__
            # As a convenience we define a default __nice__ for these objects
            return str(len(self))
        else:
            # In all other cases force the subclass to overload __nice__
            raise NotImplementedError(
                f'Define the __nice__ method for {self.__class__!r}')

    def __repr__(self):
        """str: the string of the module"""
        try:
            nice = self.__nice__()
            classname = self.__class__.__name__
            return f'<{classname}({nice}) at {hex(id(self))}>'
        except NotImplementedError as ex:
            warnings.warn(str(ex), category=RuntimeWarning)
            return object.__repr__(self)

    def __str__(self):
        """str: the string of the module"""
        try:
            classname = self.__class__.__name__
            nice = self.__nice__()
            return f'<{classname}({nice})>'
        except NotImplementedError as ex:
            warnings.warn(str(ex), category=RuntimeWarning)
            return object.__repr__(self)
        
        

class AssignResult(NiceRepr):
    """Stores assignments between predicted and truth boxes.

    Attributes:
        num_gts (int): the number of truth boxes considered when computing this
            assignment
        gt_inds (Tensor): for each predicted box indicates the 1-based
            index of the assigned truth box. 0 means unassigned and -1 means
            ignore.
        max_overlaps (Tensor): the iou between the predicted box and its
            assigned truth box.
        labels (Tensor): If specified, for each predicted box
            indicates the category label of the assigned truth box.

    Example:
        >>> # An assign result between 4 predicted boxes and 9 true boxes
        >>> # where only two boxes were assigned.
        >>> num_gts = 9
        >>> max_overlaps = torch.LongTensor([0, .5, .9, 0])
        >>> gt_inds = torch.LongTensor([-1, 1, 2, 0])
        >>> labels = torch.LongTensor([0, 3, 4, 0])
        >>> self = AssignResult(num_gts, gt_inds, max_overlaps, labels)
        >>> print(str(self))  # xdoctest: +IGNORE_WANT
        <AssignResult(num_gts=9, gt_inds.shape=(4,), max_overlaps.shape=(4,),
                      labels.shape=(4,))>
        >>> # Force addition of gt labels (when adding gt as proposals)
        >>> new_labels = torch.LongTensor([3, 4, 5])
        >>> self.add_gt_(new_labels)
        >>> print(str(self))  # xdoctest: +IGNORE_WANT
        <AssignResult(num_gts=9, gt_inds.shape=(7,), max_overlaps.shape=(7,),
                      labels.shape=(7,))>
    """

    def __init__(self, num_gts: int, gt_inds: torch.Tensor, max_overlaps: torch.Tensor,
                 labels: torch.Tensor) -> None:
        self.num_gts = num_gts
        self.gt_inds = gt_inds
        self.max_overlaps = max_overlaps
        self.labels = labels
        # Interface for possible user-defined properties
        self._extra_properties = {}

    @property
    def num_preds(self):
        """int: the number of predictions in this assignment"""
        return len(self.gt_inds)

    def set_extra_property(self, key, value):
        """Set user-defined new property."""
        assert key not in self.info
        self._extra_properties[key] = value

    def get_extra_property(self, key):
        """Get user-defined property."""
        return self._extra_properties.get(key, None)

    @property
    def info(self):
        """dict: a dictionary of info about the object"""
        basic_info = {
            'num_gts': self.num_gts,
            'num_preds': self.num_preds,
            'gt_inds': self.gt_inds,
            'max_overlaps': self.max_overlaps,
            'labels': self.labels,
        }
        basic_info.update(self._extra_properties)
        return basic_info

    def __nice__(self):
        """str: a "nice" summary string describing this assign result"""
        parts = []
        parts.append(f'num_gts={self.num_gts!r}')
        if self.gt_inds is None:
            parts.append(f'gt_inds={self.gt_inds!r}')
        else:
            parts.append(f'gt_inds.shape={tuple(self.gt_inds.shape)!r}')
        if self.max_overlaps is None:
            parts.append(f'max_overlaps={self.max_overlaps!r}')
        else:
            parts.append('max_overlaps.shape='
                         f'{tuple(self.max_overlaps.shape)!r}')
        if self.labels is None:
            parts.append(f'labels={self.labels!r}')
        else:
            parts.append(f'labels.shape={tuple(self.labels.shape)!r}')
        return ', '.join(parts)

    @classmethod
    def random(cls, num_preds=10, num_gts=5, num_classes=3, seed=None):
        """Create random AssignResult for tests or debugging.
        
        Simplified version without MMLab dependencies.

        Args:
            num_preds (int): number of predicted boxes
            num_gts (int): number of true boxes  
            num_classes (int): number of object classes
            seed (int, optional): random seed for reproducibility

        Returns:
            AssignResult: Randomly generated assign results.
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        if num_gts == 0:
            # No ground truth, all predictions are background
            gt_inds = torch.zeros(num_preds, dtype=torch.long)
            max_overlaps = torch.zeros(num_preds, dtype=torch.float32)
            labels = torch.zeros(num_preds, dtype=torch.long)
        else:
            # Randomly assign some predictions to ground truth
            gt_inds = torch.zeros(num_preds, dtype=torch.long)
            
            # Assign first min(num_preds, num_gts) predictions to ground truth
            n_assigned = min(num_preds, num_gts)
            if n_assigned > 0:
                # Randomly shuffle ground truth indices
                gt_indices = torch.randperm(num_gts)[:n_assigned] + 1  # 1-based
                pred_indices = torch.randperm(num_preds)[:n_assigned]
                gt_inds[pred_indices] = gt_indices
            
            # Create random overlaps
            max_overlaps = torch.rand(num_preds, dtype=torch.float32)
            max_overlaps[gt_inds == 0] = 0  # Background predictions have 0 overlap
            
            # Create random labels
            labels = torch.randint(0, num_classes, (num_preds,), dtype=torch.long)
            labels[gt_inds == 0] = 0  # Background label
        
        return cls(num_gts, gt_inds, max_overlaps, labels)

    def add_gt_(self, gt_labels):
        """Add ground truth as assigned results.

        Args:
            gt_labels (torch.Tensor): Labels of gt boxes
        """
        self_inds = torch.arange(
            1, len(gt_labels) + 1, dtype=torch.long, device=gt_labels.device)
        self.gt_inds = torch.cat([self_inds, self.gt_inds])

        self.max_overlaps = torch.cat(
            [self.max_overlaps.new_ones(len(gt_labels)), self.max_overlaps])

        self.labels = torch.cat([gt_labels, self.labels])
