//
//  RoundedVisualEffectView.swift
//  CalorieTracker.ai
//
//  Created by Sadi Ashraful on 06/11/2018.
//  Copyright © 2018 Sadi Ashraful. All rights reserved.
//

import UIKit

class CornerRoundedView: UIVisualEffectView {

    override func awakeFromNib() {
        self.layer.cornerRadius = 12
        self.layer.maskedCorners = [.layerMinXMinYCorner,
                                  .layerMinXMaxYCorner,
                                  .layerMaxXMinYCorner,
                                  .layerMaxXMaxYCorner]
        self.clipsToBounds = true
    }

}
