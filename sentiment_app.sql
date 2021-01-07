SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
SET AUTOCOMMIT = 0;
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `sentiment_app`
--

-- --------------------------------------------------------

--
-- Table structure for table `hashtag`
--

CREATE TABLE `hashtag` (
  `id` int(11) NOT NULL,
  `hashtag` text NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

-- --------------------------------------------------------

--
-- Table structure for table `tweet`
--

CREATE TABLE `tweet` (
  `id` int(11) NOT NULL,
  `search_val` text NOT NULL,
  `classification_result` tinyint(1) NOT NULL,
  `created_at` text,
  `tweet_id` text,
  `text` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci,
  `source` text,
  `in_reply_to_status_id` text,
  `in_reply_to_user_id` text,
  `in_reply_to_screen_name` text,
  `user_name` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci,
  `user_screen_name` text,
  `user_location` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci,
  `user_url` text,
  `user_description` text CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci,
  `user_verified` tinyint(1) DEFAULT NULL,
  `user_followers_count` int(11) DEFAULT NULL,
  `user_friends_count` int(11) DEFAULT NULL,
  `user_listed_count` int(11) DEFAULT NULL,
  `user_favourites_count` int(11) DEFAULT NULL,
  `user_created_at` text,
  `user_id` varchar(50) DEFAULT NULL,
  `user_profile_image_url_https` text,
  `coordinates_lat` text,
  `coordinates_lon` text,
  `place_country` text,
  `place_country_code` text,
  `place_full_name` text,
  `place_id` varchar(50) DEFAULT NULL,
  `place_type` varchar(30) DEFAULT NULL,
  `quoted_status_id` varchar(50) DEFAULT NULL,
  `quoted_status` text,
  `retweeted_status` text,
  `quote_count` int(11) DEFAULT NULL,
  `reply_count` int(11) DEFAULT NULL,
  `retweet_count` int(11) DEFAULT NULL,
  `favorite_count` int(11) DEFAULT NULL,
  `retweeted` tinyint(1) DEFAULT NULL,
  `entities` text,
  `lang` varchar(30) DEFAULT NULL,
  `feeds_link` text,
  `feeds_video` text,
  `feeds_image` text,
  `created_date` datetime DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Indexes for dumped tables
--

--
-- Indexes for table `hashtag`
--
ALTER TABLE `hashtag`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `tweet`
--
ALTER TABLE `tweet`
  ADD PRIMARY KEY (`id`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `hashtag`
--
ALTER TABLE `hashtag`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT for table `tweet`
--
ALTER TABLE `tweet`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
